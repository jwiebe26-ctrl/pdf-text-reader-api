"""PDF Text Reader - OCR / text core (no GUI, headless-testable).

One rule governs every coordinate in this file:

  Every word is stored ONCE as a canonical box in *displayed* page points
  (origin top-left of the page as you see it, rotation already applied).
  The on-screen highlight, the word-box overlay AND the invisible text
  layer in the exported PDF are all derived from that same box.

  PyMuPDF uses top-left origin on BOTH sides (displayed page and the
  unrotated page it reports text in), so the conversion is a pure
  rotation mapping, done only by display_rect_to_page_rect() /
  page_rect_to_display_rect(), for rotations 0/90/180/270. Those two
  functions are exact inverses and are verified by the round-trip test
  in test_core.py. (An earlier version wrongly treated the page side as
  bottom-left origin: the exported text came out vertically mirrored
  and words fell off rotated pages. That is fixed here - and the test
  now measures the exported highlight against ground truth directly
  instead of through the same mapping, so a mirrored layer cannot
  hide again.)

Why this matters: the HTML version drew the search highlight from a
different scale than the OCR boxes, so highlights landed on the wrong
spot.  That bug cannot come back here - there is only one box per word.
"""
import math
import os
import re
import shutil
import statistics

import fitz  # PyMuPDF

# ---------------------------------------------------------------- presets
PRESETS = {
    "engineering": {
        "label": "Engineering Drawing (default)",
        "dpi": 375,
        "psm": 11,          # sparse text - drawings have text scattered
        # Second pass: PSM 4 (one column, mixed sizes) catches what
        # sparse mode misses - on the test drawing PSM 11 alone missed
        # the whole title block, PSM 4 found it. Two passes are slower
        # but accuracy is the point of this preset.
        "psm_secondary": 4,
        "dictionary": False,  # read part numbers / dimensions literally
        "description": "Scanned drawings: small text, dimensions, BOM tables.",
    },
    "general": {
        "label": "General Document",
        "dpi": 300,
        "psm": 6,           # uniform block of text
        "psm_secondary": None,
        "dictionary": True,
        "description": "Letters, forms, printed pages.",
    },
    "fast": {
        "label": "Fast",
        "dpi": 200,
        "psm": 6,
        "dictionary": True,
        "psm_secondary": None,
        "description": "Quick pass; lower accuracy on small text.",
    },
}

# Tesseract locations, in order. FIRST is the copy bundled with the
# portable Windows download (tesseract/tesseract.exe next to this file),
# so Jake needs nothing installed. The rest are fallbacks.
_TESSERACT_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "tesseract", "tesseract.exe"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "tesseract", "bin", "tesseract.exe"),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
]


def find_tesseract():
    """Return a tesseract executable path, or None."""
    for cand in _TESSERACT_CANDIDATES:
        if os.path.exists(cand):
            return cand
    return shutil.which("tesseract")


def configure_tesseract():
    """Point pytesseract at Tesseract if it is not on PATH. Returns path/None."""
    import pytesseract

    path = find_tesseract()
    if path:
        pytesseract.pytesseract.tesseract_cmd = path
    return path


# ------------------------------------------------------- rotation mapping
# PyMuPDF coordinates are top-left origin on both sides: the displayed
# (rotated) page AND the unrotated page that get_text()/insert_text()
# use. So this is a pure rotation mapping - no vertical flip anywhere.
# Verified by corner mapping: e.g. at rotation 90 the unrotated
# top-left corner (0,0) is displayed at top-right (H,0).
def display_rect_to_page_rect(rect, page_width, page_height, rotation):
    """Displayed rect (x0,y0,x1,y1, top-left origin, points) -> rect in
    the UNROTATED page's coordinates (also top-left origin, points).

    page_width / page_height are the UNROTATED page size in points.
    rotation is the page rotation in degrees clockwise (0/90/180/270).
    """
    x0, y0, x1, y1 = [float(v) for v in rect]
    W, H = float(page_width), float(page_height)
    r = int(rotation) % 360
    if r == 0:
        return (x0, y0, x1, y1)
    if r == 90:
        return (y0, H - x1, y1, H - x0)
    if r == 180:
        return (W - x1, H - y1, W - x0, H - y0)
    if r == 270:
        return (W - y1, x0, W - y0, x1)
    raise ValueError("rotation must be 0/90/180/270, got %r" % rotation)


def page_rect_to_display_rect(rect, page_width, page_height, rotation):
    """Inverse of display_rect_to_page_rect."""
    u0, v0, u1, v1 = [float(v) for v in rect]
    W, H = float(page_width), float(page_height)
    r = int(rotation) % 360
    if r == 0:
        return (u0, v0, u1, v1)
    if r == 90:
        return (H - v1, u0, H - v0, u1)
    if r == 180:
        return (W - u1, H - v1, W - u0, H - v0)
    if r == 270:
        return (v0, W - u1, v1, W - u0)
    raise ValueError("rotation must be 0/90/180/270, got %r" % rotation)


# Backwards-compatible aliases (old names, same corrected mapping).
display_rect_to_pdf_rect = display_rect_to_page_rect
pdf_rect_to_display_rect = page_rect_to_display_rect


def displayed_page_size(page):
    """(width, height) in points of the page as displayed (rotation applied)."""
    W, H = page.rect.width, page.rect.height
    if int(page.rotation) % 180 == 90:
        return (H, W)
    return (W, H)


# ------------------------------------------------------------ page reading
def page_has_text(doc, page_index, min_chars=20):
    """True if the page already has a usable embedded text layer."""
    page = doc[page_index]
    text = page.get_text().strip()
    return len(text) >= min_chars


def native_words(doc, page_index):
    """Words from the embedded text layer, as canonical displayed boxes.

    Returns list of dicts: {text, box (displayed pts), conf: 100.0, source: 'native'}
    """
    page = doc[page_index]
    W, H = page.rect.width, page.rect.height
    rot = int(page.rotation) % 360
    words = []
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        if not word.strip():
            continue
        box = page_rect_to_display_rect((x0, y0, x1, y1), W, H, rot)
        words.append({"text": word, "box": box, "conf": 100.0, "source": "native"})
    return words


def render_page_image(doc, page_index, dpi):
    """Render a page to a PIL image at the given DPI, UNROTATED.

    Tesseract cannot read sideways text, so a page saved rotated 90
    degrees would OCR as almost nothing. Rendering with the derotation
    matrix gives upright text; ocr_page_image() then maps the boxes back
    through the page rotation into displayed coordinates.
    Returns (image, scale) where scale = image pixels per unrotated point."""
    import io

    from PIL import Image

    page = doc[page_index]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom) * page.derotation_matrix)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    scale = img.width / page.rect.width  # px per unrotated point
    return img, scale


def _tess_config(preset, psm=None):
    parts = ["--oem", "3", "--psm", str(psm if psm is not None else preset["psm"])]
    if not preset["dictionary"]:
        # Turn the word-list / frequency dictionaries off so part numbers
        # and dimensions (M8, 12.5, Ø12.5) are read literally, not
        # "corrected" into dictionary words.
        parts += ["-c", "load_system_dawg=0", "-c", "load_freq_dawg=0"]
    parts += ["-c", "preserve_interword_spaces=1"]
    return " ".join(parts)


def _boxes_overlap(a, b):
    """Intersection-over-union of two (x0,y0,x1,y1) boxes."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def _merge_word_lists(words):
    """Merge words from two OCR passes: if two words sit on the same spot
    (IoU > 0.5), keep the higher-confidence one; otherwise keep both."""
    merged = []
    for w in sorted(words, key=lambda z: -z["conf"]):
        if any(_boxes_overlap(w["box"], m["box"]) > 0.5 for m in merged):
            continue
        merged.append(w)
    return sorted(merged, key=lambda z: (z["box"][1], z["box"][0]))


# ------------------------------------------- engineering symbol fixes
# Tesseract's LSTM misreads two engineering symbols, on BOTH engine
# versions tested - Linux 5.3.4 AND the bundled Windows 5.4.0 model
# (same eng.traineddata behaviour, verified with the bundled model):
#   diameter Ø  -> "@"   ("Ø12.5" OCRs as "@12.5")
#   tolerance ± -> "+"   ("±0.5" OCRs as "+0.5")
# No Tesseract setting fixes it: dictionary on/off, PSM 11/6/4, OEM 1,
# char blacklist ("@" then reads as "G12.5") and whitelist all made it
# worse. So the Engineering preset corrects the two known confusions
# after OCR, in context only - the General/Fast presets are untouched.
#   "@" at the start of a dimension token ("@12.5", "@25", "@" alone)
#       -> "Ø".  "@" is never a valid leading character on a drawing.
#   "+" as a tolerance value -> "±", only when the same line says
#       TOLERANCE/TOL, or the value follows a diameter dimension and
#       no "-<number>" partner follows on that line (a real unilateral
#       tolerance is written "+0.5 -0.0", so the "-" partner vetoes it).
_DIMENSION_RE = re.compile(r"^@(\d.*)$")
_TOLERANCE_VALUE_RE = re.compile(r"^\+\d*\.?\d+$")
_NEGATIVE_VALUE_RE = re.compile(r"^-\d*\.?\d+$")


def _fix_engineering_symbols(words):
    """Correct Ø/± misreads in place (Engineering preset only).

    Words keep their boxes and confidences - only the text changes -
    so highlights, word boxes and the exported text layer all still
    share the same canonical box. Returns the same list."""
    if not words:
        return words
    heights = [w["box"][3] - w["box"][1] for w in words]
    tol = max(3.0, (statistics.median(heights) if heights else 6.0) * 0.6)
    # Group words into lines by y-centre, same tolerance as words_to_lines.
    ordered = sorted(words, key=lambda w: ((w["box"][1] + w["box"][3]) / 2,
                                            w["box"][0]))
    lines = []
    for w in ordered:
        y_mid = (w["box"][1] + w["box"][3]) / 2.0
        placed = False
        for line in lines:
            line_mid = sum((c["box"][1] + c["box"][3]) / 2.0 for c in line) / len(line)
            if abs(y_mid - line_mid) <= tol:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])
    for line in lines:
        line.sort(key=lambda w: w["box"][0])
        texts = [w["text"] for w in line]
        has_negative = any(_NEGATIVE_VALUE_RE.match(t) for t in texts)
        for i, w in enumerate(line):
            t = w["text"]
            m = _DIMENSION_RE.match(t)
            if m:
                w["text"] = "Ø" + m.group(1)
                continue
            if t == "@":
                w["text"] = "Ø"
                continue
            if _TOLERANCE_VALUE_RE.match(t) or t == "+":
                prev = texts[i - 1].upper() if i > 0 else ""
                prev_is_tol = prev.startswith("TOL")
                prev_is_diameter = prev.startswith("Ø") or prev.startswith("@")
                if (prev_is_tol or prev_is_diameter) and not has_negative:
                    w["text"] = "±" + t[1:] if t != "+" else "±"
    return words


def _thin_line_cleanup(img):
    """Gentle cleanup that keeps thin drawing lines: grayscale + a mild
    contrast stretch.  No binarise / despeckle that would eat thin lines."""
    from PIL import ImageOps

    if img.mode != "L":
        img = img.convert("L")
    return ImageOps.autocontrast(img, cutoff=1)


def _otsu_threshold(gray):
    """Otsu's threshold for a PIL grayscale image, from its 256-bin
    histogram (PIL computes the histogram in C; the 256-bin scan is
    pure Python and takes microseconds). Deliberately no numpy: the
    Windows bundle does not ship it, so ocr_core must not import it."""
    hist = gray.histogram()
    total = sum(hist)
    if total == 0:
        return 128
    sum_all = sum(i * h for i, h in enumerate(hist))
    sum_b = 0
    w_b = 0
    max_var = -1.0
    thresh = 0
    for i, h in enumerate(hist):
        w_b += h
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * h
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var:
            max_var, thresh = var, i
    return thresh


def _otsu_binarize(img):
    """Binarise a PIL image with Otsu's threshold: pixels above the
    threshold go white, the rest black. The cut is a 256-entry lookup
    table applied by PIL in C, so it is fast on full-page renders."""
    gray = img.convert("L") if img.mode != "L" else img
    t = _otsu_threshold(gray)
    lut = [0] * (t + 1) + [255] * (255 - t)
    return gray.point(lut)


# ------------------------------------------------------- deskew + borders
# Round 2 (Nov 9, 2026): two cleanups (deskew + border whitening), both free
# and both PIL + pure Python - no numpy, no OpenCV, neither is in the
# Windows bundle (site-packages is PIL + PyMuPDF + pytesseract only).
def _row_profile_score(small):
    """Sharpness of the horizontal text-line profile of a small
    grayscale image: sum of squared differences between adjacent row
    ink totals. Higher = text lines run parallel to the rows, i.e. the
    page is straight. Row sums use C-speed sum() over slices, so one
    score costs ~10 ms on a 400-px-wide image."""
    w, h = small.size
    px = list(small.getdata())
    prev = None
    total = 0.0
    for y in range(h):
        ink = w * 255 - sum(px[y * w:(y + 1) * w])
        if prev is not None:
            d = ink - prev
            total += d * d
        prev = ink
    return total


def _estimate_skew_angle(gray, max_angle=5.0):
    """Estimate the CORRECTION angle (degrees, PIL rotate convention)
    by projection-profile search on a downscaled copy: rotate the
    thumbnail through candidate angles, keep the one with the sharpest
    row profile. Measured exact (0.00 deg error) on 2/3/5 deg skews
    and 0.00 on straight, photo and tiny-text pages, at ~0.5 s per
    page. Tesseract OSD is NOT used here - it reported Rotate 0
    (conf 4.04) on a 2 deg skew, i.e. it is blind to it. Returns 0.0
    when no correction is needed."""
    from PIL import Image

    small = gray.copy()
    small.thumbnail((400, 400), Image.LANCZOS)
    best_a, best_s = 0.0, -1.0
    a = -max_angle
    while a <= max_angle + 1e-9:
        s = _row_profile_score(
            small.rotate(a, resample=Image.BICUBIC, fillcolor=255))
        if s > best_s:
            best_s, best_a = s, a
        a += 1.0
    lo, hi = best_a - 1.0, best_a + 1.0
    a = lo
    while a <= hi + 1e-9:
        s = _row_profile_score(
            small.rotate(a, resample=Image.BICUBIC, fillcolor=255))
        if s > best_s:
            best_s, best_a = s, a
        a += 0.25
    return best_a


# Only straighten when the tilt is real: below 0.5 deg Tesseract reads
# fine, and rotating costs a resample. Measured: 2 deg skew 52/54 ->
# 54/54 after deskew, 5 deg -> 53/54 (remaining miss is the O glyph,
# not the skew), straight pages estimate exactly 0.00 and are untouched.
_DESKEW_MIN_ANGLE = 0.5


def _deskew_box_back(box, img_w, img_h, angle_deg):
    """Map one axis-aligned box from a deskewed image back to the
    original image's coordinates. The image was rotated by angle_deg
    (PIL rotate: counter-clockwise visually); the inverse rotates each
    corner by -angle_deg around the image centre and returns the
    axis-aligned bounding box of the four corners. The box grows a
    little (a tilted word's bounding box is wider/taller than the
    word - at 2 deg a 100 pt word grows ~3.5 pt tall) but its position
    is exact, so highlights and the exported text layer stay on the
    word on the ORIGINAL page, which is what the viewer shows."""
    x0, y0, x1, y1 = box
    cx, cy = img_w / 2.0, img_h / 2.0
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    xs, ys = [], []
    for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        dx, dy = x - cx, y - cy
        # Inverse of PIL's counter-clockwise rotation in y-down pixel
        # coordinates: (x, y) -> (cx + dx*cos - dy*sin,
        # cy + dx*sin + dy*cos). Verified by the dot test in
        # /tmp/round2/verify_round2.py (a dot rotated +2 deg by PIL
        # maps back to its original pixel within 1 px).
        xs.append(cx + dx * cos_a - dy * sin_a)
        ys.append(cy + dx * sin_a + dy * cos_a)
    return (min(xs), min(ys), max(xs), max(ys))


def _edge_frame_means(gray, band=8):
    """(frame_mean, page_mean, dark_fraction): mean grey of the outer
    `band`-px frame vs the whole page, and the fraction of the frame's
    pixels that are dark (<128). A dark scan edge (scanner lid, page
    curl) makes the frame much darker than the page AND dark almost
    everywhere; a merely dark page (photo lighting) darkens both
    alike; a screenshot's dark UI at the edges is dark only in
    patches. The dark fraction comes free from the two histograms
    already computed here (frame histogram = whole minus inner)."""
    W, H = gray.size
    hist = gray.histogram()
    total = sum(hist)
    page_mean = sum(i * h for i, h in enumerate(hist)) / total
    inner = gray.crop((band, band, W - band, H - band))
    ih = inner.histogram()
    itotal = sum(ih)
    inner_mean = sum(i * h for i, h in enumerate(ih)) / itotal
    frame_mean = (page_mean * total - inner_mean * itotal) / (total - itotal)
    frame_pixels = total - itotal
    dark_frame = sum(hist[:128]) - sum(ih[:128])
    dark_fraction = dark_frame / frame_pixels if frame_pixels else 0.0
    return frame_mean, page_mean, dark_fraction


# Whiten the edge band only when the frame is genuinely dark AND much
# darker than the page AND dark almost everywhere (a real dark scan
# edge). Measured on the test variants: clean pages frame 250-255 /
# gap -6..+3 / dark-fraction 0.00 (never fires); photo lighting frame
# 194 / gap 3 / 0.04 (never fires); Jake's screenshot crop frame 159
# / gap 80 / 0.46 - dark UI in patches, must NOT fire (an earlier
# mean-only gate fired on it and churned 116 words via the global
# histogram operations); synthetic dark frame 60 / gap 179 / 1.00
# (fires). The band is ~1.5 pt at any render scale, so a low-res
# image does not lose a 6 pt strip: 8 px at 375 DPI, 2 px on a
# ~90 DPI screenshot crop.
_BORDER_EDGE_MAX = 200.0
_BORDER_GAP_MIN = 40.0
_BORDER_DARK_FRACTION_MIN = 0.80
_BORDER_BAND_PT = 1.5


def _whiten_dark_edges(gray, scale):
    """If the page has a dark scan edge, whiten the outer band so
    Tesseract does not try to segment the frame as content. Returns
    the (possibly new) image. Costs ~ms: two histogram means."""
    from PIL import ImageDraw

    band = max(2, int(round(_BORDER_BAND_PT * scale)))
    frame_mean, page_mean, dark_fraction = _edge_frame_means(gray, band)
    if frame_mean >= _BORDER_EDGE_MAX or \
            (page_mean - frame_mean) < _BORDER_GAP_MIN or \
            dark_fraction < _BORDER_DARK_FRACTION_MIN:
        return gray
    g = gray.copy()
    d = ImageDraw.Draw(g)
    W, H = g.size
    b = band
    d.rectangle([0, 0, W - 1, b - 1], fill=255)
    d.rectangle([0, H - b, W - 1, H - 1], fill=255)
    d.rectangle([0, 0, b - 1, H - 1], fill=255)
    d.rectangle([W - b, 0, W - 1, H - 1], fill=255)
    return g


# Fallback trigger, measured on the synthetic drawing (see
# /tmp/verify_fixes.py): clean pages score mean conf 91.5-93.1 with
# at most 5% of words under 70; the tiny-text page scores 58.8 with
# 67% low and the combo page 84.1 with 9% low. The trigger sits
# between them - mean conf below 88, or more than 7% of words under
# 70 - so clean pages never pay for the extra pass and weak pages do.
_FALLBACK_MEAN_CONF = 88.0
_FALLBACK_LOW_FRACTION = 0.07


def _result_is_weak(words):
    """True when the standard OCR passes look unreliable enough to
    justify one extra rescue pass (Otsu + 2x upscale). Needs at
    least one OCR word; a page with no words at all stays as-is
    (a genuinely blank page is not a weak result)."""
    confs = [w["conf"] for w in words if w["source"] == "ocr" and w["conf"] >= 0]
    if not confs:
        return False
    mean_conf = sum(confs) / len(confs)
    if mean_conf < _FALLBACK_MEAN_CONF:
        return True
    low = sum(1 for c in confs if c < 70)
    return low / len(confs) > _FALLBACK_LOW_FRACTION


def _ocr_passes(img, preset, scale, page, deskew_angle=0.0):
    """Run the preset's PSM passes over one prepared image. scale is
    px per unrotated point for THIS image. Returns word dicts with
    canonical displayed-point boxes, merged across passes.
    deskew_angle: if the image was deskewed by this many degrees
    (PIL rotate), each box is mapped back through the inverse
    rotation (in this image's own pixel coordinates, so it also works
    for the 2x rescue image) before the point conversion - the
    highlight and the exported layer then sit on the ORIGINAL page."""
    import pytesseract

    passes = [preset["psm"]]
    if preset.get("psm_secondary"):
        passes.append(preset["psm_secondary"])
    words = []
    for psm in passes:
        data = pytesseract.image_to_data(
            img, config=_tess_config(preset, psm),
            output_type=pytesseract.Output.DICT,
        )
        n = len(data["text"])
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue
            x, y = float(data["left"][i]), float(data["top"][i])
            w, h = float(data["width"][i]), float(data["height"][i])
            box = (x, y, x + w, y + h)
            if deskew_angle:
                box = _deskew_box_back(box, img.width, img.height,
                                       deskew_angle)
            # box is in unrotated top-left points - the same system
            # PyMuPDF reports text in - so map straight to displayed.
            box = (box[0] / scale, box[1] / scale,
                   box[2] / scale, box[3] / scale)
            if page is not None:
                W, H = page.rect.width, page.rect.height
                rot = int(page.rotation) % 360
                box = page_rect_to_display_rect(box, W, H, rot)
            words.append({"text": text, "box": box, "conf": conf, "source": "ocr"})
    return _merge_word_lists(words) if len(passes) > 1 else words


def ocr_page_image(img, preset_name, page=None):
    """OCR one rendered page image. Returns word dicts with canonical
    displayed-point boxes.

    The image is the UNROTATED page, so pixel -> point is a single
    uniform scale into unrotated points; each box is then mapped through
    the page rotation into displayed points (the same canonical box the
    viewer, the search highlight and the export all share).
    The Engineering preset runs a second PSM pass and merges the two,
    because one segmentation mode alone misses words on drawings.
    Before OCR, two cleanups run on the shared grey image (round 2):
    deskew - the tilt is estimated by projection profile (NOT Tesseract
    OSD, which is blind to small skews) and the image straightened
    when |angle| >= 0.5 deg; boxes are mapped back through the inverse
    rotation inside _ocr_passes, so highlights stay on the original
    page - and dark-edge whitening, gated to genuinely dark scan edges.
    If the standard result is weak (see _result_is_weak), one rescue
    pass runs on the Otsu-binarised image upscaled 2x and merges in -
    binarising sharpens hairline small text Tesseract otherwise reads
    as grey mush, but it slightly hurts normal-size text, so it is a
    conditional fallback, never a replacement."""
    from PIL import Image

    preset = PRESETS[preset_name]
    # scale: px per point, from the rendered size vs unrotated page size.
    # The caller passes it via the image's info dict when known; otherwise
    # derive it from dpi (render was at preset dpi).
    scale = img.info.get("px_per_point", preset["dpi"] / 72.0)
    # Keep the plain grayscale render for the cleanups and the rescue
    # pass: Otsu thresholds the raw grey levels, not the autocontrasted
    # image the standard passes use.
    gray = img.convert("L") if img.mode != "L" else img
    deskew_angle = 0.0
    angle = _estimate_skew_angle(gray)
    if abs(angle) >= _DESKEW_MIN_ANGLE:
        deskew_angle = angle
        gray = gray.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    gray = _whiten_dark_edges(gray, scale)
    words = _ocr_passes(_thin_line_cleanup(gray), preset, scale, page,
                        deskew_angle=deskew_angle)
    if _result_is_weak(words):
        rescue = _otsu_binarize(gray)
        rescue = rescue.resize((rescue.width * 2, rescue.height * 2),
                               Image.LANCZOS)
        words = _drop_same_text_duplicates(
            _merge_word_lists(
                words + _ocr_passes(rescue, preset, scale * 2, page,
                                    deskew_angle=deskew_angle)))
    if preset_name == "engineering":
        # Tesseract misreads Ø as "@" and ± as "+" - corrected in
        # context here, boxes/confidences unchanged (see
        # _fix_engineering_symbols). Other presets are untouched.
        _fix_engineering_symbols(words)
    return words


def _drop_same_text_duplicates(words):
    """Drop near-duplicate words: the exact same text whose boxes
    overlap at all or whose centres are within 4 pt. The rescue pass
    reads the same words at slightly shifted boxes, so the plain
    IoU > 0.5 merge lets them through twice - the text panel would
    show 'TOLERANCE' twice and the export would write it twice.
    Genuinely repeated words (M8, 1, 4 in the BOM) sit rows apart,
    so they never match: their boxes do not overlap and their
    centres are far apart. Comparison is on the exact text, not the
    normalised form, so 'Ø12.5' is never dropped for a rescue '12.5'
    at the same spot. Keeps the higher-confidence copy."""
    kept = []
    for w in sorted(words, key=lambda z: -z["conf"]):
        dup = False
        for m in kept:
            if w["text"] != m["text"]:
                continue
            if _boxes_overlap(w["box"], m["box"]) > 0.05:
                dup = True
                break
            cx_w = (w["box"][0] + w["box"][2]) / 2
            cy_w = (w["box"][1] + w["box"][3]) / 2
            cx_m = (m["box"][0] + m["box"][2]) / 2
            cy_m = (m["box"][1] + m["box"][3]) / 2
            if abs(cx_w - cx_m) < 4.0 and abs(cy_w - cy_m) < 4.0:
                dup = True
                break
        if not dup:
            kept.append(w)
    return sorted(kept, key=lambda z: (z["box"][1], z["box"][0]))


def read_page(doc, page_index, preset_name, progress_cb=None):
    """Read one page: OCR always runs, and its words are merged with
    the native text layer when the page has one - native wins on
    overlap (conf 100) - reporting "mixed" when OCR added words,
    "native" when every word came from the layer, "ocr" when the
    page has no text layer at all.

    The old version trusted a non-scrap native layer blindly and
    never ran OCR on it: a CAD export whose layer holds only the
    title block and a few labels returned just that layer while the
    drawing itself went unread (0 drawing words, page reported
    read). That trap is gone - OCR now runs on every page. The
    cost is OCR time on genuinely native pages, mitigated by the
    reader running pages concurrently (see app.py).
    Returns {"words": [...], "source": "native"|"ocr"|"mixed", "page": n}."""
    page = doc[page_index]
    preset = PRESETS[preset_name]
    native = native_words(doc, page_index)
    img, scale = render_page_image(doc, page_index, preset["dpi"])
    img.info["px_per_point"] = scale
    ocr_words = ocr_page_image(img, preset_name, page=page)
    if progress_cb:
        progress_cb(page_index)
    if not native:
        return {"words": ocr_words, "source": "ocr", "page": page_index}
    # The merge above drops an OCR word only when its box overlaps a
    # native word by IoU > 0.5. Tesseract also reads slivers of native
    # glyphs as extra tokens at nearly the same spot (on the test
    # page: a second "-" 1.6 pt below the real hyphen, IoU 0.04) -
    # the same-text near-duplicate drop used by the rescue pass
    # catches those, keeping the conf-100 native copy. Genuinely
    # repeated words (M8, 1, 4 in a BOM) sit rows apart and survive.
    merged = _drop_same_text_duplicates(_merge_word_lists(native + ocr_words))
    added = any(w["source"] == "ocr" for w in merged)
    return {"words": merged, "source": "mixed" if added else "native",
            "page": page_index}


# ------------------------------------------------------------ text rebuild
def words_to_lines(words):
    """Rebuild readable lines from word boxes, row by row (BOM tables on
    drawings come out as rows, not one long jumble)."""
    if not words:
        return []
    heights = [w["box"][3] - w["box"][1] for w in words]
    tol = max(3.0, (statistics.median(heights) if heights else 6.0) * 0.6)
    ordered = sorted(words, key=lambda w: (w["box"][1], w["box"][0]))
    lines, current = [], []
    for w in ordered:
        if not current:
            current = [w]
            continue
        y_mid = (w["box"][1] + w["box"][3]) / 2.0
        cur_mid = sum((c["box"][1] + c["box"][3]) / 2.0 for c in current) / len(current)
        if abs(y_mid - cur_mid) <= tol:
            current.append(w)
        else:
            lines.append(" ".join(c["text"] for c in sorted(current, key=lambda c: c["box"][0])))
            current = [w]
    if current:
        lines.append(" ".join(c["text"] for c in sorted(current, key=lambda c: c["box"][0])))
    return lines


def words_to_text(words):
    return "\n".join(words_to_lines(words))


# ------------------------------------------------------------------ search
def _norm(text):
    """Lowercase, strip punctuation - so 'M8x1.25', 'M8 x 1.25' and
    'm8' all match each other, and 'Ø12.5' matches '12.5'."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def search_words(all_pages_words, query):
    """Global search across pages. Returns list of matches:
    {page, word_index, text, box}. Multi-word queries match consecutive
    words on the same line (punctuation-tolerant)."""
    query = (query or "").strip()
    if not query:
        return []
    qparts = [_norm(p) for p in query.split()]
    qparts = [p for p in qparts if p]
    if not qparts:
        return []
    matches = []
    for page_idx, words in enumerate(all_pages_words):
        norms = [_norm(w["text"]) for w in words]
        if len(qparts) == 1:
            for i, n in enumerate(norms):
                if qparts[0] in n:
                    matches.append({"page": page_idx, "word_index": i,
                                    "text": words[i]["text"], "box": words[i]["box"]})
        else:
            for i in range(len(words) - len(qparts) + 1):
                if norms[i:i + len(qparts)] == qparts:
                    x0 = min(words[i + k]["box"][0] for k in range(len(qparts)))
                    y0 = min(words[i + k]["box"][1] for k in range(len(qparts)))
                    x1 = max(words[i + k]["box"][2] for k in range(len(qparts)))
                    y1 = max(words[i + k]["box"][3] for k in range(len(qparts)))
                    matches.append({"page": page_idx, "word_index": i,
                                    "text": " ".join(words[i + k]["text"] for k in range(len(qparts))),
                                    "box": (x0, y0, x1, y1)})
    return matches


def low_confidence_words(words, threshold=70.0):
    return [w for w in words if w["source"] == "ocr" and w["conf"] < threshold]


# ------------------------------------------------------------------ export
def _fontsize_for(word, box, font):
    """Largest font size whose rendered width fits the box width, capped
    at the box height. Keeps the reader's highlight the same width as
    the word on the page."""
    w_pt = box[2] - box[0]
    h_pt = box[3] - box[1]
    if w_pt <= 0 or h_pt <= 0:
        return 6.0
    size = h_pt
    text_w = font.text_length(word, fontsize=size)
    if text_w > 0:
        size = min(size, size * w_pt / text_w)
    return max(2.0, size)


def export_searchable_pdf(src_path, all_pages_words, out_path):
    """Build the searchable PDF: original pages copied untouched, an
    invisible text layer added in the real word positions (render mode 3).
    A word is only added when the page does not already have that exact
    word in that spot - so pages with their own text layer are left
    alone, edited words are never written twice, and exporting the same
    file twice does not duplicate the layer. (Before, a whole page was
    skipped when it had any text at all, which dropped the layer for
    every remaining word the moment one line of a scanned page had
    been edited in by hand.)
    Returns (out_path, pages_written)."""
    doc = fitz.open(src_path)
    font = fitz.Font("helv")
    written = 0
    for page_idx, words in enumerate(all_pages_words):
        page = doc[page_idx]
        # Words the page already has, as (normalised text, centre).
        # get_text("words") returns 8-tuples: (x0, y0, x1, y1, word,
        # block, line, word_no) - the word itself is item 4.
        existing = []
        try:
            for ew in page.get_text("words"):
                existing.append((_norm(ew[4]),
                                 ((ew[0] + ew[2]) / 2, (ew[1] + ew[3]) / 2)))
        except Exception:
            existing = []
        W, H = page.rect.width, page.rect.height
        rot = int(page.rotation) % 360
        for w in words:
            x0, y0, x1, y1 = display_rect_to_page_rect(w["box"], W, H, rot)
            nw = _norm(w["text"])
            if nw and any(t == nw and x0 - 2 <= ex <= x1 + 2
                          and y0 - 2 <= ey <= y1 + 2
                          for t, (ex, ey) in existing):
                continue  # already searchable in that spot
            fs = _fontsize_for(w["text"], (x0, y0, x1, y1), font)
            # Baseline so the exported span sits centred on the word box.
            # Measured with PyMuPDF: a Helvetica span's reported bbox
            # centre is 0.39*fontsize ABOVE its baseline (top-left
            # coords), so baseline = box centre + 0.39*fontsize.
            baseline_y = (y0 + y1) / 2 + 0.39 * fs
            # Centre horizontally too: the fitted text is usually
            # narrower than the OCR box, and starting it at the left
            # edge put the reader's highlight ~3.5pt left of the word.
            text_w = font.text_length(w["text"], fontsize=fs)
            x_insert = x0 + max(0.0, ((x1 - x0) - text_w) / 2)
            page.insert_text(
                fitz.Point(x_insert, baseline_y), w["text"],
                fontname="helv", fontsize=fs, render_mode=3,
            )
            written += 1
    doc.save(out_path, garbage=3, deflate=True)
    doc.close()
    return out_path, written


def validate_searchable_pdf(path, expected_words):
    """Re-open the exported PDF and check the text is actually extractable.
    Returns (ok, found_count, total_expected, missing_sample)."""
    doc = fitz.open(path)
    extracted = []
    for page in doc:
        extracted.extend(page.get_text("words"))
    doc.close()
    found_text = " ".join(w[4] for w in extracted)
    found_norm = _norm(found_text)
    missing = []
    found = 0
    for word in expected_words:
        nw = _norm(word)
        if nw and nw in found_norm:
            found += 1
        else:
            missing.append(word)
    total = len(expected_words)
    return (found == total, found, total, missing[:10])


def export_text_pdf(all_pages_words, out_path, title="PDF Text Reader export"):
    """Text-only PDF (secondary option): just the recognised words."""
    doc = fitz.open()
    for page_idx, words in enumerate(all_pages_words):
        page = doc.new_page(width=612, height=792)
        y = 54
        page.insert_text((54, 36), "%s - page %d" % (title, page_idx + 1),
                         fontname="helv", fontsize=10)
        for line in words_to_lines(words):
            if y > 750:
                page = doc.new_page(width=612, height=792)
                y = 54
            page.insert_text((54, y), line[:180], fontname="helv", fontsize=10)
            y += 14
    doc.save(out_path)
    doc.close()
    return out_path


def export_txt(all_pages_words, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for page_idx, words in enumerate(all_pages_words):
            f.write("--- Page %d ---\n" % (page_idx + 1))
            f.write(words_to_text(words))
            f.write("\n\n")
    return out_path
