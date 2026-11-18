"""Server-side reading enhancements for the PDF Text Reader API.

Wraps ocr_core.read_page() — the proven desktop pipeline, used
unchanged — with a keep-best rotation retry that only runs on weak
pages (mean OCR confidence below 70; a sideways page reads ~1/54
words). The existing deskew only fixes small tilts; this is for full
90-degree rotations. All five non-identity orientations are tried —
the three rotations plus the two diagonal flips (transpose,
transverse): a raster can arrive mirrored as well as rotated (the
sideways test file is a transpose, not a pure rotation), and a
pure-rotation retry misses that.

Speed trim (Nov 18, 2026 — the Render free host is 0.1 CPU, and the
old full-pass-per-orientation retry pushed weak pages past the web
app's timeout into the local fallback): orientations are ranked with
LIGHT probes first — one PSM pass over a 150-DPI downscale, confidence
only, no boxes — and only the best probe orientation gets one full
pipeline pass. The probe loop stops early once a probe reads genuinely
well (mean >= 85). There is no standalone Otsu candidate any more:
the standard pipeline already merges the Otsu rescue when a page is
weak, and the standalone pass cost two full-res runs per weak page
for marginal gains.

"Best" means highest mean confidence over OCR words (native-layer
words are conf 100 and identical across candidates, so they are left
out of the ranking). A candidate only replaces the current best when
its mean is better by more than MIN_IMPROVEMENT (3 points) — a
razor-thin lead is noise, and switching on it once traded 95 words
for 36 worse ones. A retry can therefore never make a page worse
than the standard pipeline by any meaningful amount. Clean pages are
untouched: no extra work runs unless the page reads weakly.

Boxes from rotated candidates are mapped back through the inverse
rotation into the original page's point coordinates, then through the
page's own rotation, so every word ends up in the same canonical
displayed-point system read_page() uses.
"""

import time

from PIL import Image

from ocr_core import (
    PRESETS,
    _drop_same_text_duplicates,
    _merge_word_lists,
    _result_is_weak,
    _tess_config,
    native_words,
    ocr_page_image,
    page_rect_to_display_rect,
    read_page,
    render_page_image,
)

# Mean OCR confidence below this triggers the rotation retry.
ROTATE_MEAN_CONF = 70.0

# A challenger only replaces the incumbent when its mean OCR
# confidence beats it by more than this. Switching on a hair (the
# screenshot crop: Otsu 61.4 vs standard 61.0) swapped 95 mostly
# low-confidence words for 36 mostly garbage ones while losing real
# words like DATE (conf 95) — noise, not an improvement. Real wins
# measured 11+ points (tiny-page Otsu 92.0 vs 80.8, sideways
# transpose 90.3 vs 49.8), so 3.0 sits well clear of noise and well
# below any genuine win.
MIN_IMPROVEMENT = 3.0


def _ocr_mean_conf(words):
    """Mean confidence over OCR words only (native words are conf 100
    and identical across candidates, so they would only mask a weak
    OCR read). None when the page has no OCR words at all."""
    confs = [w["conf"] for w in words
             if w["source"] == "ocr" and w["conf"] >= 0]
    if not confs:
        return None
    return sum(confs) / len(confs)


def _finish(native, ocr_words):
    """Merge OCR words with the native layer, exactly as read_page()
    does: native wins on overlap, near-duplicates dropped, source is
    'mixed' when OCR added words, 'native' when it added none, 'ocr'
    when the page has no text layer."""
    if not native:
        return ocr_words, "ocr"
    merged = _drop_same_text_duplicates(
        _merge_word_lists(native + ocr_words))
    added = any(w["source"] == "ocr" for w in merged)
    return merged, "mixed" if added else "native"


# Mean OCR confidence at or above this means a candidate reads
# genuinely well; the orientation loop stops early once reached.
GOOD_ENOUGH_MEAN_CONF = 85.0


def _unrotate_box(box, orientation, page_w, page_h):
    """Map a box from a re-oriented image back to the original
    unrotated page points. orientation is the variant name; page_w /
    page_h are the ORIGINAL page size in points. Forward maps
    (verified against a marker dot in each orientation):
      rot90 (90 CW):      (x', y') = (H - y, x)
      rot180:             (x', y') = (W - x, H - y)
      rot270 (90 CCW):    (x', y') = (y, W - x)
      transpose:          (x', y') = (y, x)
      transverse:         (x', y') = (H - y, W - x)
    Each inverse below is the algebraic inverse of its forward map."""
    x0, y0, x1, y1 = box
    if orientation == "rot90":
        return (y0, page_h - x1, y1, page_h - x0)
    if orientation == "rot180":
        return (page_w - x1, page_h - y1, page_w - x0, page_h - y0)
    if orientation == "rot270":
        return (page_w - y1, x0, page_w - y0, x1)
    if orientation == "transpose":
        return (y0, x0, y1, x1)
    if orientation == "transverse":
        return (page_w - y1, page_h - x1, page_w - y0, page_h - x0)
    raise ValueError("unknown orientation %r" % orientation)


# Non-identity orientations to try on weak pages, in the order tried.
# PIL transpose ops are exact (no resampling, unlike rotate()).
_ORIENTATIONS = [
    ("rot90", Image.ROTATE_270),       # 90 degrees clockwise
    ("rot180", Image.ROTATE_180),
    ("rot270", Image.ROTATE_90),       # 90 degrees counter-clockwise
    ("transpose", Image.TRANSPOSE),    # main-diagonal flip
    ("transverse", Image.TRANSVERSE),  # anti-diagonal flip
]


# Light rotation probes run at this DPI: telling orientations apart
# does not need 375 DPI, and a probe is one PSM pass over roughly a
# sixth of the pixels.
PROBE_DPI = 150


def _probe_mean(img, preset):
    """Mean OCR confidence of one quick PSM pass over a small
    grayscale image. Boxes are not needed — the probe only ranks
    orientations — so this returns (mean_conf, word_count), with
    mean_conf None when the pass found no usable words."""
    import pytesseract

    gray = img.convert("L") if img.mode != "L" else img
    data = pytesseract.image_to_data(
        gray, config=_tess_config(preset, preset["psm"]),
        output_type=pytesseract.Output.DICT,
    )
    confs = []
    n = len(data["text"])
    for i in range(n):
        if not (data["text"][i] or "").strip():
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf < 0:
            continue
        confs.append(conf)
    if not confs:
        return None, 0
    return sum(confs) / len(confs), len(confs)


def _probe_orientations(img, scale, preset_name):
    """Rank the five non-identity orientations with light probes.
    Returns [{variant, mean_conf, words, ms}] in the order tried,
    stopping early once a probe reads genuinely well (>= 85).
    scale is px per point of the full-res render."""
    preset = PRESETS[preset_name]
    factor = (PROBE_DPI / 72.0) / scale
    out = []
    for variant_name, transpose_op in _ORIENTATIONS:
        t1 = time.monotonic()
        oriented = img.transpose(transpose_op)
        small = oriented.resize(
            (max(1, round(oriented.width * factor)),
             max(1, round(oriented.height * factor))),
            Image.LANCZOS)
        mean, words = _probe_mean(small, preset)
        out.append({"variant": "probe:" + variant_name,
                    "mean_conf": round(mean, 1)
                    if mean is not None else None,
                    "words": words,
                    "ms": round((time.monotonic() - t1) * 1000)})
        if mean is not None and mean >= GOOD_ENOUGH_MEAN_CONF:
            break
    return out


def _oriented_candidate(img, scale, doc, page_index, preset_name,
                        orientation, transpose_op):
    """OCR a re-oriented copy of the rendered page and map every word
    box back to the original page's displayed points, then merge with
    the native layer as usual. ocr_page_image() runs its full pipeline
    (deskew, edge whitening, Otsu rescue) on the re-oriented pixels;
    page=None keeps the boxes in the re-oriented image's own point
    system so the inverse mapping below is exact."""
    page = doc[page_index]
    page_w, page_h = page.rect.width, page.rect.height
    rot = int(page.rotation) % 360
    oriented = img.transpose(transpose_op)
    oriented.info["px_per_point"] = scale
    words = ocr_page_image(oriented, preset_name, page=None)
    out = []
    for w in words:
        box = _unrotate_box(w["box"], orientation, page_w, page_h)
        box = page_rect_to_display_rect(box, page_w, page_h, rot)
        out.append({"text": w["text"], "box": box,
                    "conf": w["conf"], "source": w["source"]})
    return _finish(native_words(doc, page_index), out)


def read_page_enhanced(doc, page_index, preset_name):
    """Read one page with the standard pipeline plus a keep-best
    rotation retry on weak pages only. Returns read_page()'s dict plus:
      variant:    'standard' | 'rot90' | 'rot180' | 'rot270'
                  | 'transpose' | 'transverse' — which candidate won.
      candidates: [{variant, mean_conf, words, ms}] for every candidate
                  tried, in the order tried (probes are 'probe:<name>').
      ms:         total milliseconds spent on the page.
    Clean pages cost exactly one read_page() call."""
    t0 = time.monotonic()
    base = read_page(doc, page_index, preset_name)
    best_words, best_source = base["words"], base["source"]
    best_mean = _ocr_mean_conf(best_words)
    variant = "standard"
    candidates = [{"variant": "standard",
                   "mean_conf": round(best_mean, 1)
                   if best_mean is not None else None,
                   "words": len(best_words),
                   "ms": round((time.monotonic() - t0) * 1000)}]

    if _result_is_weak(base["words"]):
        # One render shared by the probes below.
        img, scale = render_page_image(
            doc, page_index, PRESETS[preset_name]["dpi"])
        img.info["px_per_point"] = scale

        # Rotation retry, only while the page still reads weakly
        # (mean OCR confidence below 70). Light probes rank the five
        # orientations cheaply; only the best probe orientation gets
        # one full pipeline pass, and it replaces the standard result
        # only when its mean beats it by more than MIN_IMPROVEMENT.
        if best_mean is not None and best_mean < ROTATE_MEAN_CONF:
            probes = _probe_orientations(img, scale, preset_name)
            candidates.extend(probes)
            best_probe = None
            for p in probes:
                if p["mean_conf"] is None:
                    continue
                if (best_probe is None
                        or p["mean_conf"] > best_probe["mean_conf"]):
                    best_probe = p
            if (best_probe is not None
                    and best_probe["mean_conf"] > best_mean + MIN_IMPROVEMENT):
                variant_name = best_probe["variant"][len("probe:"):]
                transpose_op = dict(_ORIENTATIONS)[variant_name]
                t1 = time.monotonic()
                ori_words, ori_source = _oriented_candidate(
                    img, scale, doc, page_index, preset_name,
                    variant_name, transpose_op)
                ori_mean = _ocr_mean_conf(ori_words)
                candidates.append(
                    {"variant": variant_name,
                     "mean_conf": round(ori_mean, 1)
                     if ori_mean is not None else None,
                     "words": len(ori_words),
                     "ms": round((time.monotonic() - t1) * 1000)})
                if (ori_mean is not None
                        and ori_mean > best_mean + MIN_IMPROVEMENT):
                    best_words, best_source = ori_words, ori_source
                    best_mean, variant = ori_mean, variant_name

    return {"words": best_words, "source": best_source,
            "page": base["page"], "variant": variant,
            "candidates": candidates,
            "ms": round((time.monotonic() - t0) * 1000)}
