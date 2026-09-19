"""
MetraSight — AI Legal-Metrology Inspector
Smart India Hackathon 2026 | Problem Statement SIH26034
Ministry of Consumer Affairs, Food & Public Distribution

Dual-strategy compliance pipeline:
  1. PRIMARY  — OpenCV pre-processing + Tesseract (pytesseract) OCR + RegEx/Levenshtein rule engine
  2. FALLBACK — Optional Claude Vision structural parser for fields the primary pipeline
                cannot find or is not confident about (user supplies their own Anthropic API key)

NOTE ON OCR ENGINE: this build uses Tesseract via pytesseract instead of EasyOCR/PyTorch.
EasyOCR is generally more accurate on curved/stylized packaging text, but it drags in PyTorch
(700MB+), which regularly stalls or times out on Streamlit Community Cloud's free tier. Tesseract
needs a system package (see packages.txt) but installs in well under a minute. If you have a
paid/self-hosted environment and want the more accurate EasyOCR pipeline back, ask and it can be
swapped in again.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (see requirements.txt)

IMPORTANT COMPLIANCE DISCLAIMER
--------------------------------
This tool is a hackathon prototype / decision-support aid. The numeral-height table used in the
Rule 7 (PDP) check below is entered from the author's best recollection of the Legal Metrology
(Packaged Commodities) Rules, 2011 Second Schedule and has NOT been verified against the current
official gazette notification at the time of writing. Before using this application for a real
inspection or any enforcement decision, verify every threshold in `PDP_HEIGHT_TABLE_MM2` against
the official rule text. Do not treat any verdict from this app as legal advice.
"""

import base64
import csv
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Tesseract via pytesseract — lightweight (no PyTorch), but needs the `tesseract-ocr` system
# binary installed alongside it (see packages.txt for Streamlit Community Cloud).
try:
    import pytesseract
    from pytesseract import Output as TesseractOutput
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# Anthropic SDK is optional — only needed if the user enables the Claude Vision fallback.
try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except Exception:
    ANTHROPIC_SDK_AVAILABLE = False


# ============================================================================================
# CONSTANTS — RULE ENGINE CONFIGURATION
# ============================================================================================

STANDARD_UNIT_SYMBOLS = {"g", "kg", "ml", "l", "m", "cm", "n", "pcs"}

# Non-standard tokens that must be FLAGGED as a Rule 6(1)(c) violation even though they are
# semantically the same unit. Key = normalized lowercase token seen in OCR text,
# Value = the standard symbol it should have been.
NON_STANDARD_UNIT_MAP = {
    "gms": "g", "gm": "g", "grams": "g", "gram": "g",
    "kilo": "kg", "kilos": "kg", "kgs": "kg",
    "ltr": "l", "ltrs": "l", "litre": "l", "litres": "l", "liter": "l", "liters": "l",
    "mls": "ml", "millilitre": "ml", "millilitres": "ml",
    "mtr": "m", "mtrs": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "cms": "cm", "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm",
    "newton": "n", "newtons": "n",
    "piece": "pcs", "pieces": "pcs", "pc": "pcs", "nos": "pcs", "no.": "pcs",
}

MONTH_NAMES = (
    "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    "aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)

# Rule 7 / Second Schedule — indicative numeral-height table (mm^2 area -> minimum numeral
# height in mm). SEE DISCLAIMER AT TOP OF FILE — verify against the official rule text.
PDP_HEIGHT_TABLE_MM2 = [
    (500, 1.0),
    (2500, 2.0),
    (10000, 4.0),
    (50000, 6.0),
    (float("inf"), 8.0),
]

RULE_CITATIONS = {
    "manufacturer": "Rule 6(1)(a) — Name & complete address of Manufacturer/Packer/Importer",
    "country_of_origin": "Rule 6(1)(aa) — Country of Origin (mandatory for imported goods)",
    "commodity_name": "Rule 6(1)(b) — Generic/Common name of the commodity",
    "net_quantity": "Rule 6(1)(c) — Net Quantity in statutory units",
    "mfg_date": "Rule 6(1)(d) — Month & Year of Manufacture/Packing",
    "mrp": "Rule 6(1)(e) — Maximum Retail Price, inclusive of all taxes",
    "usp": "Rule 6(1)(l) — Unit Sale Price (for >1kg/1L or multi-unit packs)",
    "consumer_care": "Rule 6(2) — Consumer Care details (name, address, phone/email)",
    "pdp_numerals": "Rule 7, Table I (Second Schedule) — PDP area vs minimum numeral height",
}

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"  # change in the sidebar if your account uses a different model string


# ============================================================================================
# DATA MODEL
# ============================================================================================

@dataclass
class FieldResult:
    field_key: str
    label: str
    status: str  # "COMPLIANT" | "NON_COMPLIANT" | "NULL"
    value: str = ""
    confidence: float = 0.0
    source: str = ""            # which uploaded image / panel the evidence came from
    detail: str = ""            # human-readable explanation
    rescan_hint: str = ""       # what to re-photograph if NULL
    rule: str = ""


@dataclass
class EvidenceStore:
    """Aggregated OCR evidence pooled across every uploaded image of one product."""
    raw_tokens: list = field(default_factory=list)   # list of dict: text, confidence, source, bbox
    full_text: str = ""
    per_image_results: dict = field(default_factory=dict)  # filename -> list of ocr rows


# ============================================================================================
# LEVENSHTEIN DISTANCE (pure python — no extra dependency needed on Streamlit Cloud)
# ============================================================================================

def levenshtein(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insert_cost = cur_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (0 if ca == cb else 1)
            cur_row[j] = min(insert_cost, delete_cost, replace_cost)
        prev_row = cur_row
    return prev_row[-1]


def fuzzy_ratio(a: str, b: str) -> float:
    """0..1 similarity based on normalized Levenshtein distance."""
    a, b = a.lower().strip(), b.lower().strip()
    max_len = max(len(a), len(b), 1)
    return 1.0 - (levenshtein(a, b) / max_len)


def fuzzy_contains(haystack: str, keywords: list, threshold: float = 0.78) -> Optional[str]:
    """Search a block of text for fuzzy matches to any keyword; return the matched keyword or None."""
    haystack_lower = haystack.lower()
    words = re.findall(r"[a-zA-Z\.]{2,}", haystack_lower)
    for kw in keywords:
        if kw.lower() in haystack_lower:
            return kw
        for w in words:
            if fuzzy_ratio(w, kw) >= threshold:
                return kw
    return None


# ============================================================================================
# IMAGE PRE-PROCESSING ENGINE (OpenCV / PIL)
# ============================================================================================

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(arr: np.ndarray) -> Image.Image:
    if len(arr.shape) == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def apply_adaptive_threshold(gray: np.ndarray, block_size: int = 25, c: int = 10) -> np.ndarray:
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10)


def auto_upscale(bgr: np.ndarray, target_min_dim: int = 1600, max_factor: float = 3.0) -> np.ndarray:
    """
    Small stamped print (batch no., MFG/EXP dates, MRP) is often only a few pixels tall in a
    photo of the whole package, well below what OCR can read reliably (~20-30px character
    height is the rough minimum). Upscaling the WHOLE image before OCR — not just cropping —
    consistently helps Tesseract pick up that fine print, at the cost of slightly slower OCR.
    Caps the scale factor so we don't blow up memory/time on an already-huge photo.
    """
    h, w = bgr.shape[:2]
    smallest_dim = min(h, w)
    if smallest_dim >= target_min_dim:
        return bgr
    factor = min(target_min_dim / max(smallest_dim, 1), max_factor)
    if factor <= 1.01:
        return bgr
    new_w, new_h = int(w * factor), int(h * factor)
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def preprocess_pipeline(
    pil_img: Image.Image,
    use_grayscale: bool = True,
    use_clahe: bool = True,
    use_denoise: bool = True,
    use_adaptive_threshold: bool = False,
    use_auto_upscale: bool = True,
) -> Image.Image:
    """
    Cleans up glossy / low-contrast / curved packaging photos before OCR.
    Returns a PIL image (RGB or L) ready for EasyOCR.
    """
    bgr = pil_to_cv2(pil_img)
    if use_auto_upscale:
        bgr = auto_upscale(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if use_grayscale else bgr

    if use_grayscale:
        if use_denoise:
            gray = denoise(gray)
        if use_clahe:
            gray = apply_clahe(gray)
        if use_adaptive_threshold:
            gray = apply_adaptive_threshold(gray)
        return cv2_to_pil(gray)

    return cv2_to_pil(gray)


# ============================================================================================
# OCR ENGINE
# ============================================================================================

@st.cache_resource(show_spinner="Checking Tesseract OCR engine...")
def get_ocr_reader():
    """
    Returns True if the Tesseract binary + pytesseract wrapper are both usable, else None.
    (Kept as a function returning a "reader" object, rather than a bare boolean, so the rest
    of the app's call sites — written for a reader-object pattern — don't need to change.)
    """
    if not TESSERACT_AVAILABLE:
        return None
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return None


def run_ocr(pil_img: Image.Image, reader) -> list:
    """
    Returns list of dicts: {text, confidence (0-1), bbox (4 points), height_px}
    Uses Tesseract's word-level output (image_to_data) so each row is one word/token,
    matching the granularity the rest of the app (rule engine, PDP height check) expects.
    """
    if not reader:
        return []
    arr = np.array(pil_img.convert("RGB"))
    data = pytesseract.image_to_data(arr, output_type=TesseractOutput.DICT)
    rows = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf_raw = data["conf"][i]
        try:
            conf_raw = float(conf_raw)
        except (TypeError, ValueError):
            conf_raw = -1.0
        if not text or conf_raw < 0:
            continue
        left, top = data["left"][i], data["top"][i]
        width, height = data["width"][i], data["height"][i]
        bbox = [
            [left, top], [left + width, top],
            [left + width, top + height], [left, top + height],
        ]
        rows.append({
            "text": text,
            "confidence": conf_raw / 100.0,
            "bbox": bbox,
            "height_px": float(height),
            "line_key": (data.get("block_num", [0]*n)[i], data.get("par_num", [0]*n)[i], data.get("line_num", [0]*n)[i]),
        })
    return rows


def pool_tokens_to_text(ocr_rows: list) -> str:
    """
    Rebuild readable text from word-level OCR tokens: words on the SAME Tesseract line are
    joined with spaces (so multi-word phrases like "inclusive of all taxes" or a manufacturer
    address stay matchable by regex), and different lines are separated by newlines.
    """
    if not ocr_rows:
        return ""
    lines = {}
    for row in ocr_rows:
        key = row.get("line_key", (0, 0, 0))
        lines.setdefault(key, []).append(row["text"])
    # preserve reading order (Tesseract's block/par/line numbering is already top-to-bottom)
    ordered_keys = sorted(lines.keys())
    return "\n".join(" ".join(lines[k]) for k in ordered_keys)


def draw_bounding_boxes(pil_img: Image.Image, ocr_rows: list, min_conf_highlight: float = 0.5) -> Image.Image:
    bgr = pil_to_cv2(pil_img)
    for row in ocr_rows:
        pts = np.array(row["bbox"], dtype=np.int32).reshape((-1, 1, 2))
        color = (0, 200, 0) if row["confidence"] >= min_conf_highlight else (0, 0, 230)
        cv2.polylines(bgr, [pts], isClosed=True, color=color, thickness=2)
    return cv2_to_pil(bgr)


# ============================================================================================
# RULE ENGINE — FIELD EXTRACTORS
# Each extractor scans the pooled evidence (raw OCR tokens across ALL uploaded images of the
# product) using RegEx first, then falls back to fuzzy/Levenshtein keyword search for anchor
# phrases. Every extractor returns (value, confidence, source_image, detail) or None.
# ============================================================================================

def _best_source_for_text(store: EvidenceStore, needle: str) -> str:
    needle_lower = needle.lower()[:25]
    for tok in store.raw_tokens:
        if needle_lower and needle_lower in tok["text"].lower():
            return tok["source"]
    return store.raw_tokens[0]["source"] if store.raw_tokens else "unknown"


def extract_country_of_origin(store: EvidenceStore) -> Optional[dict]:
    pattern = re.compile(
        r"(?:country\s+of\s+origin|made\s+in|manufactured\s+in|product\s+of|origin\s*:)\s*[:\-]?\s*([A-Za-z\s]{3,30})",
        re.IGNORECASE,
    )
    m = pattern.search(store.full_text)
    if m:
        country = m.group(1).strip().split("\n")[0][:30]
        conf = _avg_conf_near(store, m.group(0))
        return {"value": country, "confidence": conf, "source": _best_source_for_text(store, m.group(0))}
    kw = fuzzy_contains(store.full_text, ["country of origin", "made in india", "product of india"])
    if kw:
        conf = 0.45
        return {"value": kw, "confidence": conf, "source": _best_source_for_text(store, kw)}
    return None


def _clean_ws(text: str) -> str:
    """Collapse embedded newlines/extra whitespace so extracted multi-line spans display cleanly."""
    return re.sub(r"\s+", " ", text).strip()


# ============================================================================================
# SPATIAL (COLUMN-AWARE) LABEL -> VALUE MATCHING
# Real packaging very often prints statutory fields as a stamped grid — one row of labels
# (MRP / MFD / EXP / BATCH) and a separate row of values underneath, in printer/column order
# that has nothing to do with which value belongs to which label. Plain "search nearby text"
# regexes get this wrong (e.g. picking up the EXP date as if it were MFD) because they only
# look at TEXT order, not physical position. These helpers use each OCR token's bounding box
# to pair a label with the value that is actually aligned under/after it, and only fall back to
# an unlinked whole-evidence search (explicitly flagged as such) if no aligned value is found.
# ============================================================================================

def _token_x_center(t: dict) -> float:
    xs = [p[0] for p in t["bbox"]]
    return sum(xs) / len(xs)


def _token_y_center(t: dict) -> float:
    ys = [p[1] for p in t["bbox"]]
    return sum(ys) / len(ys)


def _group_lines(store: EvidenceStore) -> list:
    """Group raw tokens into their original OCR lines, tokens left-to-right, lines top-to-bottom
    within each source image."""
    groups: Dict[tuple, list] = {}
    for t in store.raw_tokens:
        key = (t.get("source"), t.get("line_key", id(t)))
        groups.setdefault(key, []).append(t)
    lines = []
    for (source, line_key), toks in groups.items():
        toks_sorted = sorted(toks, key=_token_x_center)
        y_center = sum(_token_y_center(t) for t in toks) / len(toks)
        lines.append({
            "source": source, "tokens": toks_sorted,
            "text": " ".join(t["text"] for t in toks_sorted), "y_center": y_center,
        })
    lines.sort(key=lambda l: (str(l["source"]), l["y_center"]))
    return lines


def _cluster_rows(lines: list, y_tol: float = 18.0) -> list:
    """
    Tesseract sometimes splits one visual row (e.g. a grid box with a gap between columns) into
    several separate 'line' blocks that happen to share almost the same y-position. Treating
    those as genuinely different rows breaks column-alignment matching (a value in a
    mis-split-off column can look like the ONLY candidate in its 'line' and get accepted without
    ever being compared against the real value next to the label). Merging same-source lines
    whose y-centers are within a small tolerance into one combined row fixes this at the root.
    """
    if not lines:
        return []
    rows = []
    current = [lines[0]]
    for ln in lines[1:]:
        if ln["source"] == current[-1]["source"] and abs(ln["y_center"] - current[-1]["y_center"]) <= y_tol:
            current.append(ln)
        else:
            rows.append(current)
            current = [ln]
    rows.append(current)

    merged = []
    for group in rows:
        toks = []
        for ln in group:
            toks.extend(ln["tokens"])
        toks.sort(key=_token_x_center)
        y_center = sum(_token_y_center(t) for t in toks) / len(toks)
        merged.append({"source": group[0]["source"], "tokens": toks,
                        "text": " ".join(t["text"] for t in toks), "y_center": y_center})
    return merged


def _find_value_windows(tokens: list, value_regex, max_window: int = 3) -> list:
    """Slide a 1-3 token window across a line's tokens, returning every (matched_text, x_center,
    confidence) hit of value_regex — lets a value like 'Rs.30*' or '06 / 2025' that OCR split
    into multiple tokens still be found and located spatially."""
    hits = []
    n = len(tokens)
    for start in range(n):
        for size in range(1, max_window + 1):
            end = start + size
            if end > n:
                break
            window = tokens[start:end]
            text = " ".join(t["text"] for t in window)
            m = value_regex.search(text)
            if m:
                xc = sum(_token_x_center(t) for t in window) / len(window)
                conf = sum(t["confidence"] for t in window) / len(window)
                hits.append((m.group(0), xc, conf))
    return hits


def _extract_field_near_label(store: EvidenceStore, label_regex, value_regex,
                               global_value_regex=None, max_lines_below: int = 2) -> Optional[dict]:
    """
    Column-aware label -> value extraction. Tries, in order of decreasing certainty:
      1. The value appears right after the label on the SAME OCR line (normal inline printing).
      2. The value sits in a grid row 1-2 lines BELOW the label, in the column whose x-position
         is closest to the label's own x-position (the stamped-box case).
      3. (last resort) The value pattern appears ANYWHERE else in the pooled evidence — across
         any uploaded panel — with NO positional link to the label. This is explicitly tagged as
         a cross-reference so the rule engine / UI can be honest that it's a weaker inference,
         exactly for cases where the value was photographed separately from its label.
    Returns None if nothing at all is found.
    """
    lines = _cluster_rows(_group_lines(store))
    for i, line in enumerate(lines):
        lm = label_regex.search(line["text"])
        if not lm:
            continue
        label_toks = [t for t in line["tokens"] if label_regex.search(t["text"])]
        label_x = _token_x_center(label_toks[0]) if label_toks else None

        # 1. same line, immediately after the label text
        after = line["text"][lm.end():]
        vm = value_regex.search(after)
        if vm:
            conf = _avg_conf_near(store, vm.group(0))
            return {"value": _clean_ws(vm.group(0)), "confidence": conf, "source": line["source"],
                    "note": "printed inline with its label"}

        # 2. grid row(s) below, same source image, column-aligned to the label
        for nxt in lines[i + 1: i + 1 + max_lines_below]:
            if nxt["source"] != line["source"]:
                break
            candidates = _find_value_windows(nxt["tokens"], value_regex)
            if not candidates:
                continue
            if label_x is not None and len(candidates) > 1:
                text, xc, conf = min(candidates, key=lambda c: abs(c[1] - label_x))
            else:
                text, xc, conf = candidates[0]
            return {"value": _clean_ws(text), "confidence": conf, "source": nxt["source"],
                    "note": "grid box below label, column-aligned"}

    # 3. unlinked cross-reference anywhere in the pooled evidence (any panel)
    fallback_regex = global_value_regex or value_regex
    m = fallback_regex.search(store.full_text)
    if m:
        conf = min(_avg_conf_near(store, m.group(0)), 0.65)  # capped: position isn't verified
        return {"value": _clean_ws(m.group(0)), "confidence": conf,
                "source": _best_source_for_text(store, m.group(0)),
                "note": "found elsewhere in the evidence, not directly next to its label \u2014 cross-referenced across panels, please verify placement manually"}
    return None


def extract_manufacturer(store: EvidenceStore) -> Optional[dict]:
    anchors = ["manufactured by", "marketed by", "packed by", "manufacturer", "packer", "importer", "mfd by", "mkt by"]
    anchor_pattern = re.compile(
        r"(?:manufactured\s+by|marketed\s+by|packed\s+by|mfd\s+by|mkt\s+by|manufacturer|packer|importer)\s*[:\-]?\s*",
        re.IGNORECASE,
    )
    # Fields that signal a NEW declaration has started — the address must stop before these,
    # even though it can legitimately span 2-3 physical lines of its own.
    STOP_KEYWORDS = re.compile(
        r"^\s*(country\s+of\s+origin|customer\s+care|consumer\s+care|mrp|m\.r\.p|net\s+(wt|qty|quantity)|"
        r"mfg\s*(date)?|mfd|packed\s+on|best\s+before|use\s+by|fssai|batch|email|e-?mail|usp|unit\s+sale)",
        re.IGNORECASE,
    )
    lines = store.full_text.split("\n")
    for i, line in enumerate(lines):
        m = anchor_pattern.search(line)
        if not m:
            continue
        collected = [line[m.end():]]
        # allow the address to continue onto up to 2 more lines, stopping at the next field
        for nxt in lines[i + 1: i + 3]:
            if STOP_KEYWORDS.match(nxt) or not nxt.strip():
                break
            collected.append(nxt)
        addr = _clean_ws(" ".join(collected))
        if len(addr) >= 8:
            conf = _avg_conf_near(store, line)
            return {"value": addr[:200], "confidence": conf, "source": _best_source_for_text(store, line)}
    kw = fuzzy_contains(store.full_text, anchors)
    if kw:
        return {"value": f"(anchor '{kw}' found, address text unclear)", "confidence": 0.4,
                "source": _best_source_for_text(store, kw)}
    return None


def extract_commodity_name(store: EvidenceStore) -> Optional[dict]:
    """
    Heuristic: group tokens into their original OCR lines, drop lines that look like a
    statutory field (contain digits, currency, or keywords like 'mfg'/'mrp'/'net'/'ltd'), and
    pick the tallest remaining line (product names are almost always the largest text on a
    label) — falling back to the topmost remaining line on a height tie.
    """
    if not store.raw_tokens:
        return None

    FIELD_KEYWORDS = (
        "mfg", "mfd", "mrp", "m.r.p", "net", "wt", "qty", "quantity", "rs.", "rs ", "inr", "\u20b9",
        "ltd", "pvt", "manufactur", "market", "packed", "packer", "importer", "customer", "consumer",
        "care", "email", "www", "http", "country", "origin", "date", "batch", "fssai", "license",
        "licence", "address", "pin", "code", "usp", "unit sale", "per kg", "per litre", "per g",
    )

    lines: Dict[tuple, list] = {}
    for t in store.raw_tokens:
        key = t.get("line_key", (t["source"], id(t)))  # fall back to per-token if ungrouped
        lines.setdefault(key, []).append(t)

    candidates = []
    for key, toks in lines.items():
        line_text = " ".join(t["text"] for t in toks)
        low = line_text.lower()
        if re.search(r"\d", line_text):
            continue
        if any(kw in low for kw in FIELD_KEYWORDS):
            continue
        if not re.search(r"[A-Za-z]{3,}", line_text) or len(line_text) > 60:
            continue
        avg_height = sum(t["height_px"] for t in toks) / len(toks)
        avg_conf = sum(t["confidence"] for t in toks) / len(toks)
        candidates.append((avg_height, avg_conf, line_text, toks[0]["source"]))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)  # tallest text first
    height, conf, text, source = candidates[0]
    return {"value": text.strip(), "confidence": conf, "source": source}


def extract_net_quantity(store: EvidenceStore) -> Optional[dict]:
    pattern = re.compile(
        r"(?:net\s*(?:wt|weight|qty|quantity|contents)?\s*[:\-]?\s*)?(\d+(?:\.\d+)?)\s*"
        r"(g|gm|gms|grams?|kgs?|kilos?|ml|mls|l|ltrs?|litres?|liters?|m|mtrs?|meters?|metres?|"
        r"cm|cms|centimeters?|centimetres?|n|newtons?|pcs|pc|pieces?|nos)\b",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(store.full_text))
    if not matches:
        return None
    m = matches[0]
    number, raw_unit = m.group(1), m.group(2).lower()
    conf = _avg_conf_near(store, m.group(0))

    if raw_unit in STANDARD_UNIT_SYMBOLS:
        return {
            "value": f"{number} {raw_unit}", "confidence": conf,
            "source": _best_source_for_text(store, m.group(0)),
            "violation": False, "raw_unit": raw_unit, "number": number,
        }
    normalized = NON_STANDARD_UNIT_MAP.get(raw_unit)
    return {
        "value": f"{number} {raw_unit}", "confidence": conf,
        "source": _best_source_for_text(store, m.group(0)),
        "violation": True, "raw_unit": raw_unit, "number": number,
        "expected_symbol": normalized or "?",
    }


def extract_mfg_date(store: EvidenceStore) -> Optional[dict]:
    # Deliberately does NOT match "exp"/"expiry"/"best before"/"use by" — those are a different
    # statutory date and must never be picked up as the manufacturing date.
    label_regex = re.compile(r"\bm\.?f\.?[gd]\.?\b|manufactur\w*(?!\s*by)|pkd|packed\s*(on)?|packing\s*date", re.IGNORECASE)
    value_regex = re.compile(
        rf"\b(0?[1-9]|1[0-2])\s*[\/\-\.]\s*((?:19|20)\d{{2}})\b|\b(?:{MONTH_NAMES})[\s,\.]*(?:19|20)\d{{2}}\b",
        re.IGNORECASE,
    )
    return _extract_field_near_label(store, label_regex, value_regex)


def extract_mrp(store: EvidenceStore) -> Optional[dict]:
    label_regex = re.compile(r"\bm\.?r\.?p\.?\b|maximum\s+retail\s+price", re.IGNORECASE)
    # Local (same-line / grid-column) search can be looser — spatial alignment is what
    # disambiguates it from other numbers, not the regex strictness. Guarded against matching
    # inside a longer digit run (e.g. a helpline number) via the lookaround assertions.
    local_value_regex = re.compile(
        r"(?<!\d)(?:rs\.?|inr|₹)?\s*\d{1,6}(?:[.,]\d{1,2})?\*?(?!\d)(?!-\d)", re.IGNORECASE)
    # Global (unlinked, cross-panel) fallback must be stricter — require an explicit currency
    # marker — since there is no positional evidence tying a bare number to MRP specifically.
    global_value_regex = re.compile(r"(?:₹|rs\.?|inr)\s*\d{1,6}(?:[.,]\d{1,2})?", re.IGNORECASE)

    result = _extract_field_near_label(store, label_regex, local_value_regex,
                                        global_value_regex=global_value_regex)
    if not result:
        return None
    val = result["value"]
    if not re.search(r"rs\.?|inr|₹", val, re.IGNORECASE):
        val = f"Rs. {val}"
    result["value"] = val
    tax_kw = fuzzy_contains(
        store.full_text,
        ["inclusive of all taxes", "incl. of all taxes", "incl of all taxes", "inclusive of taxes"],
        threshold=0.72,
    )
    result["has_tax_clause"] = bool(tax_kw)
    return result


def extract_usp(store: EvidenceStore) -> Optional[dict]:
    pattern = re.compile(
        r"(?:unit\s+sale\s+price|usp)\s*[:\-]?\s*(?:rs\.?|inr|₹)?\s*(\d+(?:[.,]\d{1,2})?)\s*(?:per|/)\s*"
        r"(kg|g|l|ml|litre|liter|gram|piece|pcs)?",
        re.IGNORECASE,
    )
    m = pattern.search(store.full_text)
    if m:
        conf = _avg_conf_near(store, m.group(0))
        return {"value": _clean_ws(m.group(0)), "confidence": conf, "source": _best_source_for_text(store, m.group(0))}
    return None


def extract_consumer_care(store: EvidenceStore) -> Optional[dict]:
    phone_pattern = re.compile(r"(?:\+91[\-\s]?)?\b[6-9]\d{9}\b|\b1800[\-\s]?\d{2,3}[\-\s]?\d{3,4}\b")
    email_pattern = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    care_kw = fuzzy_contains(store.full_text, ["consumer care", "customer care", "customer support", "care of consumer"])

    phone = phone_pattern.search(store.full_text)
    email = email_pattern.search(store.full_text)

    if not (phone or email or care_kw):
        return None

    parts = []
    conf_components = []
    if care_kw:
        parts.append(f"anchor:'{care_kw}'")
        conf_components.append(0.5)
    if phone:
        parts.append(f"phone:{phone.group(0)}")
        conf_components.append(_avg_conf_near(store, phone.group(0)))
    if email:
        parts.append(f"email:{email.group(0)}")
        conf_components.append(_avg_conf_near(store, email.group(0)))

    conf = sum(conf_components) / len(conf_components) if conf_components else 0.4
    source_text = phone.group(0) if phone else (email.group(0) if email else care_kw)
    return {"value": " | ".join(parts), "confidence": conf, "source": _best_source_for_text(store, source_text),
            "has_phone": bool(phone), "has_email": bool(email)}


def _avg_conf_near(store: EvidenceStore, snippet: str) -> float:
    """Approximate confidence for a regex match by averaging confidences of OCR tokens whose
    text overlaps with the matched snippet. Falls back to the mean confidence of all tokens."""
    snippet_lower = snippet.lower()
    hits = [t["confidence"] for t in store.raw_tokens if t["text"] and t["text"].lower() in snippet_lower
            or snippet_lower[:12] in t["text"].lower()]
    if hits:
        return sum(hits) / len(hits)
    if store.raw_tokens:
        return sum(t["confidence"] for t in store.raw_tokens) / len(store.raw_tokens)
    return 0.5


# ============================================================================================
# PDP / RULE 7 NUMERAL HEIGHT CHECK
# ============================================================================================

def required_numeral_height_mm(pdp_area_cm2: float) -> float:
    area_mm2 = pdp_area_cm2 * 100.0
    for threshold, height in PDP_HEIGHT_TABLE_MM2:
        if area_mm2 <= threshold:
            return height
    return PDP_HEIGHT_TABLE_MM2[-1][1]


def check_pdp_numerals(store: EvidenceStore, pdp_area_cm2: float, px_per_mm: float) -> FieldResult:
    required_mm = required_numeral_height_mm(pdp_area_cm2)
    if px_per_mm <= 0 or not store.raw_tokens:
        return FieldResult(
            field_key="pdp_numerals", label="PDP Numeral Height", status="NULL",
            detail="No calibration (px/mm) or no OCR tokens available to measure numeral height.",
            rescan_hint="Provide a pixels-per-mm calibration value and ensure at least one image has readable text.",
            rule=RULE_CITATIONS["pdp_numerals"],
        )

    # Only consider tokens that look like the numerals used for MRP / Net Qty declarations
    numeral_tokens = [t for t in store.raw_tokens if re.search(r"\d", t["text"])]
    if not numeral_tokens:
        return FieldResult(
            field_key="pdp_numerals", label="PDP Numeral Height", status="NULL",
            detail="No numeral text detected on the Principal Display Panel images.",
            rescan_hint="Upload a sharper, well-lit close-up of the PDP showing MRP / Net Quantity numerals.",
            rule=RULE_CITATIONS["pdp_numerals"],
        )

    heights_mm = [t["height_px"] / px_per_mm for t in numeral_tokens]
    min_found = min(heights_mm)
    avg_conf = sum(t["confidence"] for t in numeral_tokens) / len(numeral_tokens)

    if min_found + 1e-6 >= required_mm:
        return FieldResult(
            field_key="pdp_numerals", label="PDP Numeral Height", status="COMPLIANT",
            value=f"{min_found:.2f} mm (min required {required_mm:.1f} mm)",
            confidence=avg_conf, source="calculated from OCR bounding boxes",
            detail=f"Smallest detected numeral height {min_found:.2f} mm meets the {required_mm:.1f} mm minimum "
                   f"for a PDP area of {pdp_area_cm2:.1f} cm².",
            rule=RULE_CITATIONS["pdp_numerals"],
        )
    return FieldResult(
        field_key="pdp_numerals", label="PDP Numeral Height", status="NON_COMPLIANT",
        value=f"{min_found:.2f} mm (min required {required_mm:.1f} mm)",
        confidence=avg_conf, source="calculated from OCR bounding boxes",
        detail=f"Smallest detected numeral height {min_found:.2f} mm is BELOW the {required_mm:.1f} mm minimum "
               f"required for a PDP area of {pdp_area_cm2:.1f} cm².",
        rule=RULE_CITATIONS["pdp_numerals"],
    )


# ============================================================================================
# CLAUDE VISION FALLBACK (optional, user-supplied API key)
# ============================================================================================

FALLBACK_FIELD_PROMPTS = {
    "manufacturer": "the full name and complete address of the manufacturer, packer, or importer",
    "country_of_origin": "the declared country of origin",
    "commodity_name": "the generic/common name of the product",
    "net_quantity": "the net quantity value and its unit exactly as printed",
    "mfg_date": "the month and year of manufacture or packing",
    "mrp": "the Maximum Retail Price figure and whether an 'inclusive of all taxes' phrase appears near it",
    "usp": "the Unit Sale Price if printed (price per kg/litre/unit)",
    "consumer_care": "any consumer/customer care name, address, phone number, or email",
}


def call_claude_vision_fallback(api_key: str, model: str, image_bytes: bytes, missing_field_key: str) -> Optional[dict]:
    """
    Sends ONE image to Claude with a vision request asking specifically for the missing field.
    Returns {"found": bool, "value": str, "confidence_hint": "high"/"medium"/"low"} or None on error.
    This is a best-effort structural parser used only when the primary CV+OCR pipeline could not
    find a mandatory field with sufficient confidence.
    """
    if not ANTHROPIC_SDK_AVAILABLE:
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        field_desc = FALLBACK_FIELD_PROMPTS.get(missing_field_key, missing_field_key)
        prompt = (
            f"You are assisting a Legal Metrology compliance check on a packaged commodity label photo. "
            f"Look ONLY for {field_desc}. "
            f"Respond with STRICT JSON only, no markdown fences, no preamble, in this exact shape: "
            f'{{"found": true or false, "value": "<the exact text you see, or empty string>", '
            f'"confidence_hint": "high" or "medium" or "low"}}. '
            f"If you cannot clearly see this information on the label, set found to false."
        )
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text_out = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        cleaned = re.sub(r"^```(?:json)?|```$", "", text_out.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        return parsed
    except Exception as e:
        return {"found": False, "value": "", "confidence_hint": "low", "error": str(e)}


# ============================================================================================
# CONFIDENCE / NULL PROTOCOL — MERGES PRIMARY + FALLBACK INTO FieldResult LIST
# ============================================================================================

RESCAN_HINTS = {
    "manufacturer": "Manufacturer/packer address block unclear. Upload a clear close-up of the back panel address text.",
    "country_of_origin": "Country of Origin not found. Upload a close-up of the panel stating 'Country of Origin' or 'Made in ___'.",
    "commodity_name": "Product/common name unclear. Upload a straight-on, well-lit photo of the front panel.",
    "net_quantity": "Net Quantity block unclear. Upload a close-up of the weight/volume declaration.",
    "mfg_date": "Manufacturing date stamp unclear. Upload a close-up of the embossed/printed date stamp.",
    "mrp": "MRP block unclear. Please upload a clear close-up of the price/MRP stamp.",
    "usp": "Unit Sale Price not found (required for packs >1kg/1L or multi-unit packs). Upload a close-up of the pricing panel.",
    "consumer_care": "Consumer care details unclear. Upload a close-up of the back panel with contact information.",
}


def evaluate_all_rules(store: EvidenceStore, conf_threshold: float, net_qty_extra: Optional[dict] = None) -> list:
    results = []

    def make_result(key, label, extractor_result, ok_check=lambda r: True, extra_detail=""):
        if extractor_result is None or extractor_result.get("confidence", 0) < conf_threshold:
            return FieldResult(
                field_key=key, label=label, status="NULL",
                value=extractor_result["value"] if extractor_result else "",
                confidence=extractor_result["confidence"] if extractor_result else 0.0,
                detail="Field not found or below the confidence threshold across all uploaded images.",
                rescan_hint=RESCAN_HINTS.get(key, "Please upload a clearer image of this section."),
                rule=RULE_CITATIONS[key],
            )
        compliant = ok_check(extractor_result)
        note = extractor_result.get("note", "")
        detail = f"{extra_detail} ({note})".strip() if extra_detail and note else (note or extra_detail)
        return FieldResult(
            field_key=key, label=label,
            status="COMPLIANT" if compliant else "NON_COMPLIANT",
            value=extractor_result["value"], confidence=extractor_result["confidence"],
            source=extractor_result.get("source", ""), detail=detail,
            rule=RULE_CITATIONS[key],
        )

    results.append(make_result("manufacturer", "Manufacturer / Packer / Importer", extract_manufacturer(store)))
    results.append(make_result("country_of_origin", "Country of Origin", extract_country_of_origin(store)))
    results.append(make_result("commodity_name", "Commodity / Product Name", extract_commodity_name(store)))

    nq = extract_net_quantity(store)
    if nq is None or nq.get("confidence", 0) < conf_threshold:
        results.append(FieldResult(
            field_key="net_quantity", label="Net Quantity", status="NULL",
            value=nq["value"] if nq else "", confidence=nq["confidence"] if nq else 0.0,
            detail="Net quantity declaration not found or below confidence threshold.",
            rescan_hint=RESCAN_HINTS["net_quantity"], rule=RULE_CITATIONS["net_quantity"],
        ))
    else:
        if nq["violation"]:
            results.append(FieldResult(
                field_key="net_quantity", label="Net Quantity", status="NON_COMPLIANT",
                value=nq["value"], confidence=nq["confidence"], source=nq["source"],
                detail=f"Non-standard unit symbol '{nq['raw_unit']}' used. Statutory symbol is "
                       f"'{nq['expected_symbol']}'. Only standard symbols (g, kg, ml, l, m, cm, N, Pcs) are permitted.",
                rule=RULE_CITATIONS["net_quantity"],
            ))
        else:
            results.append(FieldResult(
                field_key="net_quantity", label="Net Quantity", status="COMPLIANT",
                value=nq["value"], confidence=nq["confidence"], source=nq["source"],
                detail="Statutory unit symbol used correctly.", rule=RULE_CITATIONS["net_quantity"],
            ))

    results.append(make_result("mfg_date", "Month & Year of Manufacture/Packing", extract_mfg_date(store)))

    mrp = extract_mrp(store)
    if mrp is None or mrp.get("confidence", 0) < conf_threshold:
        results.append(FieldResult(
            field_key="mrp", label="Maximum Retail Price (MRP)", status="NULL",
            value=mrp["value"] if mrp else "", confidence=mrp["confidence"] if mrp else 0.0,
            detail="MRP not found or below confidence threshold.",
            rescan_hint=RESCAN_HINTS["mrp"], rule=RULE_CITATIONS["mrp"],
        ))
    else:
        note_suffix = f" ({mrp['note']})" if mrp.get("note") else ""
        if mrp["has_tax_clause"]:
            results.append(FieldResult(
                field_key="mrp", label="Maximum Retail Price (MRP)", status="COMPLIANT",
                value=mrp["value"], confidence=mrp["confidence"], source=mrp["source"],
                detail=f"MRP found together with an 'inclusive of all taxes' declaration.{note_suffix}",
                rule=RULE_CITATIONS["mrp"],
            ))
        else:
            results.append(FieldResult(
                field_key="mrp", label="Maximum Retail Price (MRP)", status="NON_COMPLIANT",
                value=mrp["value"], confidence=mrp["confidence"], source=mrp["source"],
                detail="MRP found, but no 'inclusive of all taxes' / 'incl. of all taxes' phrase was detected "
                       f"nearby, which Rule 6(1)(e) requires.{note_suffix}",
                rule=RULE_CITATIONS["mrp"],
            ))

    # USP is conditional — only mandatory if net quantity > 1 kg / 1 L or the pack is a multi-unit pack.
    usp_required = False
    if nq and not nq.get("violation") and nq.get("raw_unit") in ("kg", "l"):
        try:
            usp_required = float(nq["number"]) > 1.0
        except ValueError:
            usp_required = False
    usp = extract_usp(store)
    if usp_required:
        if usp is None or usp.get("confidence", 0) < conf_threshold:
            results.append(FieldResult(
                field_key="usp", label="Unit Sale Price (USP)", status="NULL",
                value=usp["value"] if usp else "", confidence=usp["confidence"] if usp else 0.0,
                detail="Pack exceeds 1 kg/1 L so USP is mandatory, but it was not found.",
                rescan_hint=RESCAN_HINTS["usp"], rule=RULE_CITATIONS["usp"],
            ))
        else:
            results.append(FieldResult(
                field_key="usp", label="Unit Sale Price (USP)", status="COMPLIANT",
                value=usp["value"], confidence=usp["confidence"], source=usp["source"],
                detail="USP declaration found for a multi-unit / >1kg/1L pack.", rule=RULE_CITATIONS["usp"],
            ))
    else:
        results.append(FieldResult(
            field_key="usp", label="Unit Sale Price (USP)", status="COMPLIANT",
            value=usp["value"] if usp else "Not applicable (pack ≤ 1kg/1L)",
            confidence=1.0 if not usp else usp["confidence"],
            detail="USP is not mandatory for this pack size.", rule=RULE_CITATIONS["usp"],
        ))

    cc = extract_consumer_care(store)
    if cc is None or cc.get("confidence", 0) < conf_threshold:
        results.append(FieldResult(
            field_key="consumer_care", label="Consumer Care Details", status="NULL",
            value=cc["value"] if cc else "", confidence=cc["confidence"] if cc else 0.0,
            detail="Consumer care name/address/phone/email not found or below confidence threshold.",
            rescan_hint=RESCAN_HINTS["consumer_care"], rule=RULE_CITATIONS["consumer_care"],
        ))
    else:
        if cc["has_phone"] or cc["has_email"]:
            results.append(FieldResult(
                field_key="consumer_care", label="Consumer Care Details", status="COMPLIANT",
                value=cc["value"], confidence=cc["confidence"], source=cc["source"],
                detail="At least one verifiable contact channel (phone or email) found.", rule=RULE_CITATIONS["consumer_care"],
            ))
        else:
            results.append(FieldResult(
                field_key="consumer_care", label="Consumer Care Details", status="NON_COMPLIANT",
                value=cc["value"], confidence=cc["confidence"], source=cc["source"],
                detail="A 'consumer care' anchor phrase was found but no verifiable phone number or email "
                       "was detected nearby.", rule=RULE_CITATIONS["consumer_care"],
            ))

    return results


# ============================================================================================
# REPORT GENERATION
# ============================================================================================

STATUS_ICON = {"COMPLIANT": "🟢", "NON_COMPLIANT": "🔴", "NULL": "🟡"}


def build_report_text(product_name: str, results: list, pdp_result: FieldResult) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("METRASIGHT — DIGITAL LEGAL METROLOGY INSPECTION REPORT")
    lines.append("Legal Metrology (Packaged Commodities) Rules, 2011")
    lines.append("=" * 78)
    lines.append(f"Product: {product_name or 'Unnamed Product'}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("-" * 78)
    all_results = results + [pdp_result]
    n_compliant = sum(1 for r in all_results if r.status == "COMPLIANT")
    n_noncompliant = sum(1 for r in all_results if r.status == "NON_COMPLIANT")
    n_null = sum(1 for r in all_results if r.status == "NULL")
    lines.append(f"Summary: {n_compliant} Compliant | {n_noncompliant} Non-Compliant | {n_null} Null/Re-scan needed")
    lines.append("-" * 78)
    for r in all_results:
        lines.append("")
        lines.append(f"{STATUS_ICON[r.status]} [{r.status}] {r.label}")
        lines.append(f"   Rule: {r.rule}")
        lines.append(f"   Detected Value: {r.value or '(none)'}")
        lines.append(f"   Confidence: {r.confidence:.0%}")
        if r.source:
            lines.append(f"   Source: {r.source}")
        if r.detail:
            lines.append(f"   Detail: {r.detail}")
        if r.status == "NULL" and r.rescan_hint:
            lines.append(f"   ⚠ EVIDENCE PLANNER ALERT: {r.rescan_hint}")
    lines.append("")
    lines.append("=" * 78)
    lines.append("This report is generated by an automated prototype tool (SIH26034 MetraSight) and")
    lines.append("does not constitute a formal legal determination. Fields marked NULL require manual")
    lines.append("re-inspection before any enforcement action.")
    lines.append("=" * 78)
    return "\n".join(lines)


def build_report_csv(product_name: str, results: list, pdp_result: FieldResult) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Product", product_name or "Unnamed Product"])
    writer.writerow(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow([])
    writer.writerow(["Rule Citation", "Field", "Status", "Detected Value", "Confidence", "Source", "Detail", "Re-scan Hint"])
    for r in results + [pdp_result]:
        writer.writerow([r.rule, r.label, r.status, r.value, f"{r.confidence:.2f}", r.source, r.detail, r.rescan_hint])
    return output.getvalue()


# ============================================================================================
# STREAMLIT APP
# ============================================================================================

st.set_page_config(page_title="MetraSight | AI Legal-Metrology Inspector", page_icon="⚖️", layout="wide")

CUSTOM_CSS = """
<style>
.main { background-color: #0e1117; }
.metrasight-header {
    background: linear-gradient(90deg, #0b3d91 0%, #1b5fae 60%, #12805c 100%);
    padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1.2rem;
}
.metrasight-header h1 { color: white; margin: 0; font-size: 1.9rem; }
.metrasight-header p { color: #dbe9ff; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
.compliance-card {
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.card-compliant { background-color: rgba(19, 128, 92, 0.12); border-left: 5px solid #13805c; }
.card-noncompliant { background-color: rgba(200, 40, 40, 0.10); border-left: 5px solid #c82828; }
.card-null { background-color: rgba(210, 160, 20, 0.12); border-left: 5px solid #d2a014; }
.card-title { font-weight: 700; font-size: 1.02rem; margin-bottom: 0.25rem; }
.card-rule { font-size: 0.78rem; opacity: 0.75; margin-bottom: 0.4rem; }
.card-value { font-size: 0.92rem; margin-bottom: 0.2rem; }
.alert-box {
    background-color: rgba(210, 160, 20, 0.18); border: 1px dashed #d2a014;
    border-radius: 8px; padding: 0.6rem 0.9rem; margin-top: 0.4rem; font-size: 0.85rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="metrasight-header">
        <h1>⚖️ MetraSight — AI Legal-Metrology Inspector</h1>
        <p>SIH26034 · Ministry of Consumer Affairs, Food & Public Distribution ·
        Legal Metrology (Packaged Commodities) Rules, 2011 compliance engine</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not TESSERACT_AVAILABLE:
    st.error(
        "pytesseract is not installed, or the `tesseract-ocr` system package is missing. "
        "Check that both `requirements.txt` (pytesseract) and `packages.txt` (tesseract-ocr) are present "
        "at the root of your repo, then reboot the app."
    )

# ---------------------------- SESSION STATE ----------------------------
if "evidence_store" not in st.session_state:
    st.session_state.evidence_store = EvidenceStore()
if "processed_images" not in st.session_state:
    st.session_state.processed_images = {}   # filename -> PIL image with boxes
if "raw_ocr_by_image" not in st.session_state:
    st.session_state.raw_ocr_by_image = {}
if "rule_results" not in st.session_state:
    st.session_state.rule_results = []
if "pdp_result" not in st.session_state:
    st.session_state.pdp_result = None

# ---------------------------- SIDEBAR ----------------------------
with st.sidebar:
    st.header("⚙️ Inspection Settings")
    product_name = st.text_input("Product Name (for the report header)", value="")

    st.subheader("Pre-processing")
    use_auto_upscale = st.checkbox(
        "Auto-upscale small print", value=True,
        help="Boosts resolution before OCR so tiny stamped text (MRP/MFG/EXP/Batch box) is tall "
             "enough for Tesseract to read. Strongly recommended — leave this on.",
    )
    use_grayscale = st.checkbox("Grayscale conversion", value=True)
    use_clahe = st.checkbox("CLAHE contrast stretching", value=True)
    use_denoise = st.checkbox("Denoise (reduces glare speckle)", value=True)
    use_adaptive_threshold = st.checkbox("Adaptive thresholding", value=False)

    st.subheader("Confidence & PDP")
    conf_threshold = st.slider("Minimum field confidence to accept (%)", 0, 100, 50) / 100.0
    pdp_area_cm2 = st.slider("Principal Display Panel (PDP) area (cm²)", 1.0, 3000.0, 100.0)
    px_per_mm = st.number_input(
        "Calibration: pixels per millimetre in the uploaded photo",
        min_value=0.0, value=10.0, step=0.5,
        help="Needed to convert detected text height in pixels to real-world mm for the Rule 7 numeral-height check. "
             "Estimate by measuring a known-size reference object in the photo (e.g. a 1 cm grid or ruler).",
    )

    st.subheader("Fallback: Claude Vision Parser")
    enable_fallback = st.checkbox("Enable Claude Vision fallback for NULL fields", value=False)
    claude_api_key = ""
    claude_model = DEFAULT_CLAUDE_MODEL
    if enable_fallback:
        if not ANTHROPIC_SDK_AVAILABLE:
            st.warning("The `anthropic` package is not installed. Add it to requirements.txt to use this feature.")
        claude_api_key = st.text_input("Your Anthropic API key", type="password",
                                        help="Your key is used only for this session and is never stored.")
        claude_model = st.text_input(
            "Claude model string", value=DEFAULT_CLAUDE_MODEL,
            help="Check docs.claude.com for the current list of available model strings for your account.",
        )

    st.markdown("---")
    st.caption(
        "⚠️ The Rule 7 numeral-height table used in this prototype is an indicative approximation. "
        "Verify against the official Second Schedule before relying on this tool for real inspections."
    )

# ---------------------------- TABS ----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Upload & Pre-process",
    "2️⃣ OCR Evidence Store",
    "3️⃣ Compliance Report",
    "4️⃣ Download Audit Report",
])

# ---------------------------- TAB 1: UPLOAD & PREPROCESS ----------------------------
with tab1:
    st.subheader("Multi-Image Upload")
    st.write("Upload every available panel of the SAME product (front, back, top/bottom, close-ups). "
             "MetraSight pools evidence across all images before evaluating compliance.")
    uploaded_files = st.file_uploader(
        "Upload product images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
    )

    run_button = st.button("🔍 Run Pre-processing + OCR on all images", type="primary", disabled=not uploaded_files)

    if run_button and uploaded_files:
        reader = get_ocr_reader()
        if reader is None:
            st.error("Tesseract OCR could not be loaded. Check requirements.txt / packages.txt / installation logs.")
        else:
            store = EvidenceStore()
            processed_images = {}
            raw_ocr_by_image = {}
            progress = st.progress(0.0, text="Starting...")
            for i, uf in enumerate(uploaded_files):
                progress.progress((i) / len(uploaded_files), text=f"Processing {uf.name}...")
                pil_img = Image.open(uf)
                pre_img = preprocess_pipeline(
                    pil_img, use_grayscale, use_clahe, use_denoise, use_adaptive_threshold, use_auto_upscale
                )
                ocr_rows = run_ocr(pre_img, reader)
                for row in ocr_rows:
                    row["source"] = uf.name
                    store.raw_tokens.append(row)
                store.full_text += "\n" + pool_tokens_to_text(ocr_rows)
                raw_ocr_by_image[uf.name] = ocr_rows

                boxed_display = draw_bounding_boxes(pil_img.convert("RGB"), ocr_rows, conf_threshold)
                processed_images[uf.name] = {"preprocessed": pre_img, "boxed": boxed_display}

            progress.progress(1.0, text="Done.")
            st.session_state.evidence_store = store
            st.session_state.processed_images = processed_images
            st.session_state.raw_ocr_by_image = raw_ocr_by_image
            st.success(
                f"Processed {len(uploaded_files)} image(s) — {len(store.raw_tokens)} text regions detected. "
                f"Go to Tab 2 to review evidence, or Tab 3 for the compliance report."
            )

    if st.session_state.processed_images:
        st.markdown("---")
        st.subheader("Preview: Pre-processed Images & Detected Text Regions")
        cols = st.columns(2)
        for idx, (fname, imgs) in enumerate(st.session_state.processed_images.items()):
            with cols[idx % 2]:
                st.markdown(f"**{fname}**")
                st.image(imgs["boxed"], caption="Bounding boxes (green = high confidence, red = low)", use_container_width=True)
                with st.expander("View pre-processed (cleaned) image"):
                    st.image(imgs["preprocessed"], use_container_width=True)

# ---------------------------- TAB 2: OCR EVIDENCE STORE ----------------------------
with tab2:
    st.subheader("Aggregated Raw OCR Output")
    store: EvidenceStore = st.session_state.evidence_store
    if not store.raw_tokens:
        st.info("No OCR evidence yet. Upload images and run pre-processing + OCR in Tab 1.")
    else:
        st.code(store.full_text.strip() or "(empty)", language="text")

        st.subheader("Per-token Detail (all uploaded images combined)")
        df = pd.DataFrame([
            {"Text": t["text"], "Confidence": round(t["confidence"], 2), "Height (px)": round(t["height_px"], 1),
             "Source Image": t["source"]}
            for t in store.raw_tokens
        ])
        st.dataframe(df, use_container_width=True, height=320)

        st.subheader("Fuzzy Token Matches (unit / anchor phrase detection)")
        fuzzy_hits = []
        for kw in ["consumer care", "customer care", "manufactured by", "marketed by", "packed by",
                   "country of origin", "made in", "mrp", "inclusive of all taxes"]:
            match = fuzzy_contains(store.full_text, [kw], threshold=0.72)
            if match:
                fuzzy_hits.append({"Anchor Phrase": kw, "Matched In Text": "Yes"})
        unit_hits = []
        for tok in store.raw_tokens:
            for bad_unit, correct in NON_STANDARD_UNIT_MAP.items():
                if re.search(rf"\b{re.escape(bad_unit)}\b", tok["text"], re.IGNORECASE):
                    unit_hits.append({"Non-standard Unit Found": bad_unit, "Should Be": correct, "Source": tok["source"]})
        if fuzzy_hits:
            st.dataframe(pd.DataFrame(fuzzy_hits), use_container_width=True)
        if unit_hits:
            st.warning("Non-standard unit tokens detected (Rule 6(1)(c) risk):")
            st.dataframe(pd.DataFrame(unit_hits), use_container_width=True)
        if not fuzzy_hits and not unit_hits:
            st.caption("No fuzzy anchor phrases or non-standard units detected.")

# ---------------------------- TAB 3: COMPLIANCE REPORT ----------------------------
with tab3:
    st.subheader("Structured Compliance Verdicts")
    store: EvidenceStore = st.session_state.evidence_store
    if not store.raw_tokens:
        st.info("No evidence to evaluate yet. Upload images and run OCR in Tab 1.")
    else:
        if st.button("🧮 Evaluate Compliance Rules", type="primary"):
            results = evaluate_all_rules(store, conf_threshold)
            pdp_result = check_pdp_numerals(store, pdp_area_cm2, px_per_mm)

            # ---- OPTIONAL FALLBACK: Claude Vision for any NULL mandatory field ----
            if enable_fallback and claude_api_key and ANTHROPIC_SDK_AVAILABLE:
                null_fields = [r for r in results if r.status == "NULL"]
                if null_fields and st.session_state.processed_images:
                    with st.spinner(f"Running Claude Vision fallback for {len(null_fields)} field(s)..."):
                        for r in null_fields:
                            for fname in st.session_state.processed_images:
                                try:
                                    buf = io.BytesIO()
                                    st.session_state.processed_images[fname]["preprocessed"].convert("RGB").save(buf, format="JPEG")
                                    fb = call_claude_vision_fallback(claude_api_key, claude_model, buf.getvalue(), r.field_key)
                                except Exception as e:
                                    fb = {"found": False, "error": str(e)}
                                if fb and fb.get("found"):
                                    r.value = fb.get("value", r.value)
                                    hint = fb.get("confidence_hint", "medium")
                                    r.confidence = {"high": 0.85, "medium": 0.65, "low": 0.45}.get(hint, 0.5)
                                    r.source = f"Claude Vision / {fname}"
                                    if r.field_key == "net_quantity":
                                        # Vision fallback only recovers the raw text; it does not re-run the
                                        # statutory-unit RegEx check, so flag this for manual unit verification.
                                        r.status = "NON_COMPLIANT"
                                        r.detail = (f"Recovered via Claude Vision fallback (image: {fname}), but the "
                                                    f"statutory-unit-symbol check could not be re-run automatically — "
                                                    f"please manually verify the unit symbol is one of g, kg, ml, l, m, cm, N, Pcs.")
                                    else:
                                        r.status = "COMPLIANT"
                                        r.detail = f"Recovered via Claude Vision fallback parser (image: {fname})."
                                    break

            st.session_state.rule_results = results
            st.session_state.pdp_result = pdp_result

        if st.session_state.rule_results:
            results = st.session_state.rule_results
            pdp_result = st.session_state.pdp_result
            all_results = results + ([pdp_result] if pdp_result else [])

            n_compliant = sum(1 for r in all_results if r.status == "COMPLIANT")
            n_noncompliant = sum(1 for r in all_results if r.status == "NON_COMPLIANT")
            n_null = sum(1 for r in all_results if r.status == "NULL")
            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 Compliant", n_compliant)
            c2.metric("🔴 Non-Compliant", n_noncompliant)
            c3.metric("🟡 Null / Re-scan Needed", n_null)

            st.markdown("---")
            for r in all_results:
                css_class = {"COMPLIANT": "card-compliant", "NON_COMPLIANT": "card-noncompliant", "NULL": "card-null"}[r.status]
                icon = STATUS_ICON[r.status]
                html = f"""
                <div class="compliance-card {css_class}">
                    <div class="card-title">{icon} {r.label} — {r.status.replace('_',' ')}</div>
                    <div class="card-rule">{r.rule}</div>
                    <div class="card-value"><b>Detected:</b> {r.value or '(none)'} &nbsp;|&nbsp; <b>Confidence:</b> {r.confidence:.0%}
                    {'&nbsp;|&nbsp; <b>Source:</b> ' + r.source if r.source else ''}</div>
                    {'<div class="card-value">' + r.detail + '</div>' if r.detail else ''}
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
                if r.status == "NULL" and r.rescan_hint:
                    st.markdown(f'<div class="alert-box">📸 <b>Evidence Planner Alert:</b> {r.rescan_hint}</div>',
                                unsafe_allow_html=True)
        else:
            st.caption("Click 'Evaluate Compliance Rules' to generate verdicts from the pooled evidence.")

# ---------------------------- TAB 4: DOWNLOAD AUDIT REPORT ----------------------------
with tab4:
    st.subheader("Downloadable Official Inspection Report")
    if not st.session_state.rule_results:
        st.info("Run the compliance evaluation in Tab 3 first.")
    else:
        results = st.session_state.rule_results
        pdp_result = st.session_state.pdp_result
        report_txt = build_report_text(product_name, results, pdp_result)
        report_csv = build_report_csv(product_name, results, pdp_result)

        st.text_area("Report Preview", report_txt, height=400)

        colA, colB = st.columns(2)
        with colA:
            st.download_button(
                "⬇️ Download Report (.txt)", data=report_txt,
                file_name=f"metrasight_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )
        with colB:
            st.download_button(
                "⬇️ Download Report (.csv)", data=report_csv,
                file_name=f"metrasight_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption(
    "MetraSight prototype for SIH26034. Automated verdicts are decision-support only and must be "
    "confirmed by a human inspector before any regulatory action, especially any field the tool "
    "returns as NULL / Re-scan needed."
)
