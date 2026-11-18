"""PDF Text Reader API — server-side OCR for the web version.

The browser uploads a PDF (or image); this API reads it with the
same pipeline as the desktop app (ocr_core.read_page, plus the
keep-best retries in ocr_enhanced), returns the words as JSON, and
deletes the upload. Nothing is ever stored: no database, no disk
retention, no file names or content in the logs.

Endpoints:
  GET  /health  -> {"ok": true, "tesseract": "<version>",
                    "tmp_files": <files in the temp dir>}
  POST /ocr     -> multipart "file" (PDF or image), form fields
                   "preset" (engineering|general|fast, default
                   engineering) and "pages" ("all" or "1-3" or "2",
                   1-based, default "all").
                   Returns {"pages": [{"page", "source", "words",
                   "mean_conf", "variant", "ms"}], "words_total",
                   "preset", "pages_total"}.
                   Word boxes are in PDF points, the same coordinate
                   system the desktop app uses. Pages are processed
                   ONE at a time, sequentially, to fit the free tier
                   (512 MB RAM, shared CPU) — never hold all pages
                   in memory at once.

Privacy: each upload lives in its own temp subdir and is deleted in
a finally block before the response is sent, along with anything
else created for the request. Uploads are capped (default 100 MB).
"""

import os
import re
import shutil
import tempfile
import time

# Single-threaded tesseract measured faster than multi-threaded on
# shared-CPU hosts, and avoids oversubscription (the desktop pool
# only behaved with it). Child tesseract processes inherit this.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

from flask import Flask, jsonify, request

import fitz  # PyMuPDF

from ocr_core import PRESETS, configure_tesseract
from ocr_enhanced import read_page_enhanced

app = Flask(__name__)

# Upload cap: 100 MB by default (PTR_MAX_MB overrides, mainly so the
# rejection path can be tested without a 100 MB file). Flask answers
# oversize uploads with 413 before the handler runs.
MAX_MB = int(os.environ.get("PTR_MAX_MB", "100"))
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

# One temp root for all uploads, so /health can count leftovers.
# Each request gets its own subdir, deleted in a finally block.
TMP_ROOT = os.environ.get(
    "PTR_TMPDIR", os.path.join(tempfile.gettempdir(), "ptr-ocr"))
os.makedirs(TMP_ROOT, exist_ok=True)

ALLOWED_ORIGINS = {"https://pdf-text-reader-web.onrender.com"}
LOCALHOST_RE = re.compile(r"^http://(localhost|127\.0\.0\.1):\d+$")

ALLOWED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".webp",
                ".tif", ".tiff", ".bmp"}


def _origin_allowed(origin):
    return origin in ALLOWED_ORIGINS or bool(
        origin and LOCALHOST_RE.match(origin))


@app.after_request
def _cors(response):
    origin = request.headers.get("Origin")
    if origin and _origin_allowed(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    return response


@app.route("/ocr", methods=["OPTIONS"])
def _ocr_preflight():
    origin = request.headers.get("Origin")
    if not origin or not _origin_allowed(origin):
        return jsonify({"error": "origin not allowed"}), 403
    resp = jsonify({})
    resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Max-Age"] = "86400"
    resp.headers["Vary"] = "Origin"
    return resp, 204


def _tesseract_version():
    try:
        import pytesseract
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return "unknown"


def _tmp_file_count():
    count = 0
    for _root, _dirs, files in os.walk(TMP_ROOT):
        count += len(files)
    return count


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True,
                    "tesseract": _tesseract_version(),
                    "tmp_files": _tmp_file_count()})


def _parse_pages(raw, page_count):
    """'all' -> every page; '2' -> page 2; '1-3' -> pages 1..3.
    Page numbers are 1-based in the request, 0-based internally."""
    if raw is None or raw.strip().lower() == "all":
        return list(range(page_count))
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", raw.strip())
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if start < 1 or end < start or end > page_count:
            raise ValueError("pages range out of bounds")
        return list(range(start - 1, end))
    m = re.fullmatch(r"(\d+)", raw.strip())
    if m:
        page = int(m.group(1))
        if page < 1 or page > page_count:
            raise ValueError("page out of bounds")
        return [page - 1]
    raise ValueError("pages must be 'all', a page number, or a range")


def _word_json(w):
    x0, y0, x1, y1 = w["box"]
    return {"text": w["text"],
            "x0": round(float(x0), 2), "y0": round(float(y0), 2),
            "x1": round(float(x1), 2), "y1": round(float(y1), 2),
            "conf": round(float(w["conf"]), 1)}


def _save_upload(tmp_path, data, ext):
    """Write the upload to tmp_path as a PDF the reader can open.

    PDFs are written as-is. Images are wrapped in a one-page PDF
    sized so the image's own pixels map to ~300 DPI on the page:
    PyMuPDF would otherwise size an image page at 96 DPI, so the
    reader's 375 DPI render would upscale the image ~4x beyond its
    own pixels — pure cost (a screenshot crop took 153s that way)
    with no extra information. Very large images are also downscaled
    to 5000 px on the long side first, so one upload cannot blow
    the 512 MB free tier with a gigapixel render."""
    if ext == ".pdf" or data[:5] == b"%PDF-":
        with open(tmp_path, "wb") as f:
            f.write(data)
        return
    import io
    from PIL import Image as PILImage
    try:
        pil = PILImage.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise ValueError("could not read file as PDF or image")
    if max(pil.size) > 5000:
        pil.thumbnail((5000, 5000), PILImage.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, "PNG")
    doc = fitz.open()
    page = doc.new_page(width=pil.width * 72 / 300,
                        height=pil.height * 72 / 300)
    page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(tmp_path)
    doc.close()


@app.route("/ocr", methods=["POST"])
def ocr():
    started = time.monotonic()
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    upload = request.files["file"]
    if not upload or not upload.filename:
        return jsonify({"error": "empty file"}), 400
    ext = os.path.splitext(upload.filename)[1].lower()
    if ext and ext not in ALLOWED_EXTS:
        return jsonify({"error": "unsupported file type"}), 400

    preset = (request.form.get("preset") or "engineering").strip()
    if preset not in PRESETS:
        return jsonify({"error": "preset must be engineering, "
                                 "general or fast"}), 400

    data = upload.read()
    if not data:
        return jsonify({"error": "empty file"}), 400

    # Each request gets its own temp subdir; everything in it is
    # deleted in the finally block before the response is sent.
    req_dir = tempfile.mkdtemp(prefix="ocr-", dir=TMP_ROOT)
    try:
        tmp_path = os.path.join(req_dir, "upload.pdf")
        try:
            _save_upload(tmp_path, data, ext)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        try:
            doc = fitz.open(tmp_path)
        except Exception:
            return jsonify({"error": "could not read file as "
                                     "PDF or image"}), 400
        with doc:
            page_count = doc.page_count
            try:
                page_indexes = _parse_pages(
                    request.form.get("pages"), page_count)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            pages_out = []
            words_total = 0
            for idx in page_indexes:
                result = read_page_enhanced(doc, idx, preset)
                words = result["words"]
                mean_conf = (round(sum(w["conf"] for w in words)
                                   / len(words), 1) if words else 0.0)
                pages_out.append({
                    "page": idx + 1,  # 1-based in the response
                    "source": result["source"],
                    "words": [_word_json(w) for w in words],
                    "mean_conf": mean_conf,
                    "variant": result["variant"],
                    "ms": result["ms"],
                })
                words_total += len(words)
        app.logger.info("ocr preset=%s pages=%d words=%d ms=%d",
                        preset, len(pages_out), words_total,
                        round((time.monotonic() - started) * 1000))
        return jsonify({"pages": pages_out,
                        "words_total": words_total,
                        "preset": preset,
                        "pages_total": page_count})
    finally:
        shutil.rmtree(req_dir, ignore_errors=True)


@app.errorhandler(400)
def _bad_request(e):
    return jsonify({"error": "bad request"}), 400


@app.errorhandler(404)
def _not_found(e):
    return jsonify({"error": "not found"}), 404


@app.errorhandler(413)
def _too_large(e):
    return jsonify({"error": "file too large (max %d MB)" % MAX_MB}), 413


@app.errorhandler(500)
def _server_error(e):
    return jsonify({"error": "internal error"}), 500


if __name__ == "__main__":
    configure_tesseract()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
