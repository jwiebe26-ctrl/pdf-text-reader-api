# PDF Text Reader API

Server-side OCR for the [PDF Text Reader](https://pdf-text-reader-web.onrender.com)
web app. The browser uploads a PDF or image, this API reads it with the
same pipeline as the desktop app, returns the words as JSON, and deletes
the upload. Nothing is ever stored: no database, no disk retention, no
file names or content in the logs.

## How it reads

`ocr_core.py` is copied verbatim from the desktop app: Engineering
preset (~375 DPI, PSM 11 + PSM 4 merged, dictionary off, Ø/±
post-corrections, Otsu rescue, deskew, border whitening, native-layer
merge with native winning on overlap), tessdata_fast model.

`ocr_enhanced.py` adds two keep-best retries that only run on weak
pages, so clean pages cost exactly one standard read:

- **Otsu standalone variant** — when a page reads weakly (mean conf
  < 88 or > 7% of words below 70, the same gate as the desktop rescue),
  the Otsu-binarized 2x-upscale image is tried as its own candidate and
  the better result wins by mean OCR confidence.
- **Auto-rotate retry** — when mean OCR confidence is below 70 (a
  sideways page reads ~1/54 words), OCR is re-run on 90/180/270-degree
  rotated copies and the best result wins. The existing deskew only
  fixes small tilts; this is for full 90-degree rotations.

Pages are processed one at a time, sequentially — the free tier is
512 MB RAM on shared CPU, and a worker pool oversubscribes small hosts.
Tesseract runs single-threaded (`OMP_THREAD_LIMIT=1`), which measured
faster than multi-threaded on shared CPUs.

## Endpoints

- `GET /health` → `{"ok": true, "tesseract": "5.3.4", "tmp_files": 0}`
- `POST /ocr` — multipart `file` (PDF or image, max 100 MB), form
  fields `preset` (`engineering` | `general` | `fast`, default
  `engineering`) and `pages` (`all` or `1-3` or `2`, 1-based, default
  `all`). Returns:
  ```json
  {"pages": [{"page": 1, "source": "mixed",
              "words": [{"text": "M8", "x0": 10.5, "y0": 20.1,
                         "x1": 30.2, "y1": 32.4, "conf": 93.0}],
              "mean_conf": 91.2, "variant": "standard", "ms": 3200}],
   "words_total": 54, "preset": "engineering", "pages_total": 1}
  ```
  Word boxes are in PDF points, the same coordinate system the desktop
  app uses. `variant` says which candidate won (`standard`, `otsu`,
  `rot90`, `rot180`, `rot270`, `transpose`, `transverse`). `ms` is the
  milliseconds spent on that page.

CORS allows only `https://pdf-text-reader-web.onrender.com` (and
`http://localhost:*` for testing).

## Deploy (Render, free)

1. Push this repo to GitHub.
2. Render dashboard → **New +** → **Web Service** → connect the repo.
3. Render detects `render.yaml`: Docker environment, **Free** plan.
   (If it asks, confirm the plan is Free — never a paid plan.)
4. Wait for the first deploy, then check
   `https://<your-service>.onrender.com/health` — it should return
   `{"ok": true, ...}`.
5. Point the web app at the service URL. Free services sleep after
   ~15 minutes idle; the first request after sleeping takes a few
   seconds to wake up.

## Run locally

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# Tesseract 5 + the fast English model:
#   macOS: brew install tesseract
#   Ubuntu: sudo apt-get install tesseract-ocr
#   then download eng.traineddata from
#   https://github.com/tesseract-ocr/tessdata_fast
#   and set TESSDATA_PREFIX to its folder.
PORT=5000 python app.py
curl http://localhost:5000/health
curl -F file=@drawing.pdf -F preset=engineering \
     -F pages=all http://localhost:5000/ocr
```

## Privacy

Uploads live in a per-request temp subdir under `PTR_TMPDIR`
(default: the system temp dir) and are deleted in a `finally` block
before the response is sent, along with anything else created for the
request. The server logs only method, preset, page/word counts and
timing — never file names or content.
