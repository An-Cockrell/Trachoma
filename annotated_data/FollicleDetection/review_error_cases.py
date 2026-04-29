#!/usr/bin/env python3
"""
Visualizer for false positive / false negative error cases.

Usage:
    python review_error_cases.py <txt_file> <masks_dir>

Example:
    python review_error_cases.py \\
        follicle_detection_figs/old_script_SAMFollicleMask_TF_eval/false_negatives.txt \\
        follicle_detection_figs/old_script_SAMFollicleMask_TF_eval/fn_masks/

Open http://localhost:5000

Controls:
    g          — toggle mask overlay on/off
    ← / →      — previous / next image
"""

import sys
import os
import io

import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, send_file

app   = Flask(__name__)
STATE = {}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_pairs(txt_file, masks_dir):
    """
    Read image paths from txt_file, pair each with its mask in masks_dir.
    Skips any entry whose mask file doesn't exist.
    """
    with open(txt_file) as f:
        img_paths = [line.strip() for line in f if line.strip()]

    pairs = []
    for img_path in img_paths:
        stem      = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(masks_dir, stem + ".png")
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
        else:
            print(f"Warning: no mask found for {img_path}, skipping.")
    return pairs


def mask_to_rgba_png_bytes(mask_u8, img_path):
    """
    Resize binary mask (0/255) to match the original image dimensions,
    then return as RGBA PNG bytes with a red overlay.
    """
    orig      = cv2.imread(img_path)
    H, W      = orig.shape[:2]
    resized   = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
    rgba      = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[resized > 0] = [255, 50, 50, 160]
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    buf.seek(0)
    return buf


# ── HTML ───────────────────────────────────────────────────────────────────────

HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Error Case Reviewer</title>
<style>
  * { box-sizing: border-box; }
  body   { font-family: sans-serif; background: #1a1a1a; color: #eee;
           text-align: center; margin: 0; padding: 20px; user-select: none; }
  #info  { font-size: .9em; color: #aaa; margin-bottom: 4px; }
  #fname { font-size: 1em; color: #ccc; margin-bottom: 8px; word-break: break-all; }
  #status { margin-bottom: 10px; font-size: .9em;
            display: flex; gap: 20px; justify-content: center; align-items: center; }
  .badge     { padding: 2px 10px; border-radius: 4px; font-weight: bold; }
  .badge-on  { background: #2980b9; color: #fff; }
  .badge-off { background: #555;    color: #aaa; }
  canvas { display: block; margin: 0 auto;
           max-width: 900px; max-height: 700px;
           border: 2px solid #444; }
  #btns { margin-top: 14px; display: flex; gap: 30px; justify-content: center; }
  button { padding: 10px 36px; font-size: 1em; font-weight: bold;
           border: none; border-radius: 6px; cursor: pointer;
           background: #444; color: #eee; }
  button:hover { background: #666; }
  #keys { font-size: .75em; color: #666; margin-top: 12px; }
  #done { font-size: 1.4em; margin-top: 40px; color: #4CAF50; display: none; }
</style>
</head>
<body>
<div id="info">Loading…</div>
<div id="fname"></div>
<div id="status">
  Mask: <span id="mask-badge" class="badge badge-on">ON</span>
</div>
<canvas id="c"></canvas>
<div id="btns">
  <button onclick="navigate('prev')">&#9664; Prev</button>
  <button onclick="navigate('next')">Next &#9654;</button>
</div>
<div id="keys">g: toggle mask &nbsp;|&nbsp; ← →: navigate</div>
<div id="done"></div>

<script>
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
let baseImg  = null;
let maskImg  = null;
let showMask = true;

function loadImg(src) {
  return new Promise((res, rej) => {
    const i = new Image();
    i.onload  = () => res(i);
    i.onerror = rej;
    i.src = src;
  });
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (baseImg) ctx.drawImage(baseImg, 0, 0);
  if (showMask && maskImg) ctx.drawImage(maskImg, 0, 0);
}

function updateMaskUI() {
  const badge = document.getElementById('mask-badge');
  badge.textContent = showMask ? 'ON' : 'OFF';
  badge.className   = showMask ? 'badge badge-on' : 'badge badge-off';
}

async function load() {
  const d = await fetch('/current').then(r => r.json());
  if (d.done) {
    canvas.style.display = 'none';
    document.getElementById('btns').style.display = 'none';
    document.getElementById('done').style.display = 'block';
    document.getElementById('done').textContent   = d.message;
    document.getElementById('info').textContent   = '';
    return;
  }
  document.getElementById('info').textContent  = (d.idx + 1) + ' / ' + d.total;
  document.getElementById('fname').textContent = d.filename;

  const [img, msk] = await Promise.all([
    loadImg('/file?p=' + encodeURIComponent(d.image_path) + '&t=' + Date.now()),
    loadImg('/mask_png?t=' + Date.now()),
  ]);
  baseImg        = img;
  maskImg        = msk;
  canvas.width   = img.naturalWidth;
  canvas.height  = img.naturalHeight;
  redraw();
}

async function navigate(dir) {
  await fetch('/navigate', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ dir }),
  });
  load();
}

document.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
  switch (e.key) {
    case 'g':
      showMask = !showMask;
      updateMaskUI();
      redraw();
      break;
    case 'ArrowLeft':  navigate('prev'); break;
    case 'ArrowRight': navigate('next'); break;
  }
});

load();
</script>
</body>
</html>
"""


# ── Flask routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML


@app.route("/current")
def current():
    s   = STATE
    idx = s["idx"]
    if not s["pairs"]:
        return jsonify({"done": True, "message": "No pairs loaded."})
    img_path, _ = s["pairs"][idx]
    return jsonify({
        "done":       False,
        "idx":        idx,
        "total":      len(s["pairs"]),
        "filename":   os.path.basename(img_path),
        "image_path": img_path,
    })


@app.route("/file")
def serve_file():
    path = request.args.get("p", "")
    if not os.path.isfile(path):
        return "not found", 404
    return send_file(path)


@app.route("/mask_png")
def mask_png():
    s = STATE
    img_path, mask_path = s["pairs"][s["idx"]]
    mask_u8 = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return send_file(mask_to_rgba_png_bytes(mask_u8, img_path), mimetype="image/png")


@app.route("/navigate", methods=["POST"])
def navigate():
    direction = request.get_json().get("dir")
    s         = STATE
    if direction == "prev":
        s["idx"] = max(0, s["idx"] - 1)
    else:
        s["idx"] = min(len(s["pairs"]) - 1, s["idx"] + 1)
    return jsonify({"status": "ok"})


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python review_error_cases.py <txt_file> <masks_dir>")
        sys.exit(1)

    txt_file  = sys.argv[1]
    masks_dir = sys.argv[2]

    if not os.path.isfile(txt_file):
        print(f"Error: txt file not found: {txt_file}")
        sys.exit(1)
    if not os.path.isdir(masks_dir):
        print(f"Error: masks directory not found: {masks_dir}")
        sys.exit(1)

    pairs = load_pairs(txt_file, masks_dir)
    if not pairs:
        print("No valid image/mask pairs found.")
        sys.exit(1)

    STATE.update({"pairs": pairs, "idx": 0})

    print(f"Loaded {len(pairs)} cases from {txt_file}")
    print("Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
