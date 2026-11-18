"""Server-side reading enhancements for the PDF Text Reader API.

Wraps ocr_core.read_page() — the proven desktop pipeline, used
unchanged — with two keep-best retries that only run on weak pages:

  1. Otsu standalone variant: the Otsu-binarized 2x-upscale image run
     as its OWN candidate. The desktop pipeline already has this rescue
     inside ocr_page_image(), merged into the standard result; here it
     is also tried standalone, and whichever result (standard or Otsu)
     has the better mean OCR confidence wins. Pages that already read
     well never see it.
  2. Auto-rotate retry: when a page's mean OCR confidence is below 70
     (a sideways page reads ~1/54 words), re-run OCR on the other
     orientations of the rendered page and keep the best by mean OCR
     confidence. The existing deskew only fixes small tilts; this is
     for full 90-degree rotations. All five non-identity orientations
     are tried — the three rotations plus the two diagonal flips
     (transpose, transverse): a raster can arrive mirrored as well as
     rotated (the sideways test file is a transpose, not a pure
     rotation), and a pure-rotation retry misses that. The loop stops
     early once a candidate reads genuinely well (mean >= 85), so a
     normal upside-down or sideways page pays for one or two extra
     passes, not five.

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
    _DESKEW_MIN_ANGLE,
    _drop_same_text_duplicates,
    _estimate_skew_angle,
    _fix_engineering_symbols,
    _merge_word_lists,
    _ocr_passes,
    _otsu_binarize,
    _result_is_weak,
    _whiten_dark_edges,
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


def _otsu_candidate(img, scale, doc, page_index, preset_name):
    """The Otsu-binarized 2x-upscale image as a standalone candidate.
    Same preprocessing and rescue branch ocr_page_image() uses
    (deskew, dark-edge whitening, Otsu, LANCZOS 2x), run on its own
    instead of merged into the standard result."""
    preset = PRESETS[preset_name]
    page = doc[page_index]
    gray = img.convert("L") if img.mode != "L" else img
    deskew_angle = 0.0
    angle = _estimate_skew_angle(gray)
    if abs(angle) >= _DESKEW_MIN_ANGLE:
        deskew_angle = angle
        gray = gray.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    gray = _whiten_dark_edges(gray, scale)
    rescue = _otsu_binarize(gray)
    rescue = rescue.resize((rescue.width * 2, rescue.height * 2),
                           Image.LANCZOS)
    words = _ocr_passes(rescue, preset, scale * 2, page,
                        deskew_angle=deskew_angle)
    if preset_name == "engineering":
        _fix_engineering_symbols(words)
    return _finish(native_words(doc, page_index), words)


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
    """Read one page with the standard pipeline plus keep-best retries
    on weak pages only. Returns read_page()'s dict plus:
      variant:    'standard' | 'otsu' | 'rot90' | 'rot180' | 'rot270'
                  | 'transpose' | 'transverse' — which candidate won.
      candidates: [{variant, mean_conf, words, ms}] for every candidate
                  tried, in the order tried.
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
        # One render shared by every extra candidate below.
        img, scale = render_page_image(
            doc, page_index, PRESETS[preset_name]["dpi"])
        img.info["px_per_point"] = scale

        # Candidate 2: Otsu standalone. Only replaces the standard
        # result when its mean OCR confidence is strictly better.
        t1 = time.monotonic()
        otsu_words, otsu_source = _otsu_candidate(
            img, scale, doc, page_index, preset_name)
        otsu_mean = _ocr_mean_conf(otsu_words)
        candidates.append({"variant": "otsu",
                           "mean_conf": round(otsu_mean, 1)
                           if otsu_mean is not None else None,
                           "words": len(otsu_words),
                           "ms": round((time.monotonic() - t1) * 1000)})
        if (otsu_mean is not None and best_mean is not None
                and otsu_mean > best_mean + MIN_IMPROVEMENT):
            best_words, best_source = otsu_words, otsu_source
            best_mean, variant = otsu_mean, "otsu"

        # Candidates 3+: other orientations, only while the best so
        # far is still weak (mean OCR confidence below 70). A page
        # the Otsu variant already fixed does not pay for more passes.
        # The loop also stops early once a candidate reads genuinely
        # well (mean >= 85) — a later orientation cannot beat a good
        # read by enough to be worth another full pass.
        if best_mean is not None and best_mean < ROTATE_MEAN_CONF:
            for variant_name, transpose_op in _ORIENTATIONS:
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
                if (ori_mean is not None and best_mean is not None
                        and ori_mean > best_mean + MIN_IMPROVEMENT):
                    best_words, best_source = ori_words, ori_source
                    best_mean, variant = ori_mean, variant_name
                if best_mean is not None and best_mean >= GOOD_ENOUGH_MEAN_CONF:
                    break

    return {"words": best_words, "source": best_source,
            "page": base["page"], "variant": variant,
            "candidates": candidates,
            "ms": round((time.monotonic() - t0) * 1000)}
