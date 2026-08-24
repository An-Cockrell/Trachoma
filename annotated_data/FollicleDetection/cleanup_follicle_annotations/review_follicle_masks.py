#!/usr/bin/env python3
"""
Review app for SAM3 follicle mask pairs with interactive mask editing.

Usage: python review_follicle_masks.py [parent_dir]
Default parent_dir: manual_check (relative to this script)
Open http://localhost:5000

Controls:
  g         — toggle mask overlay on/off
  e         — delete mode: click a follicle blob to remove it
  a         — add mode: click+drag to draw a new follicle circle
  k         — keep pair (saves modified mask to disk)
  d         — discard pair
"""

import sys
import os
import io
import csv
import itertools

import cv2 as cv
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, send_file

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
KEEP_CSV          = "keep.csv"
CLEANED_CSV       = "cleaned.csv"
DISCARD_CSV       = "discard.csv"
STARRED_CSV       = "starred.csv"
CSV_HEADER        = ["image_path", "mask_path"]
MIN_FOLLICLE_RATIO = 0.0012  # min_follicle_area_px = ratio * eyelid_area_px at 512x512

MAX_DISPLAY_LONG_EDGE = 3000  # pixels; images larger than this are downscaled for display only

app   = Flask(__name__)
STATE = {}


def compute_ref_radius(img_path, display_size):
    """Radius in display pixels of the minimum detectable follicle.

    Counts non-black pixels in the 512x512 image as the eyelid area, applies
    MIN_FOLLICLE_RATIO to get the minimum follicle area at that scale, then
    scales the resulting radius to display coordinates.
    """
    with Image.open(img_path) as im:
        arr = np.asarray(im.convert("RGB").resize((512, 512), Image.LANCZOS))
    eyelid_area = int(np.any(arr > 10, axis=2).sum())
    ref_r_512   = np.sqrt(MIN_FOLLICLE_RATIO * eyelid_area / np.pi)
    dW, dH      = display_size
    scale       = np.sqrt((dW / 512) * (dH / 512))
    return float(ref_r_512 * scale)


def compute_display_size(W, H):
    """Return (display_W, display_H) capped at MAX_DISPLAY_LONG_EDGE on the long edge."""
    long_edge = max(W, H)
    if long_edge <= MAX_DISPLAY_LONG_EDGE:
        return W, H
    scale = MAX_DISPLAY_LONG_EDGE / long_edge
    return round(W * scale), round(H * scale)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_starred(parent_dir):
    starred = set()
    fpath = os.path.join(parent_dir, STARRED_CSV)
    if os.path.exists(fpath):
        with open(fpath, newline="") as f:
            for row in csv.DictReader(f):
                starred.add(row["image_path"])
    return starred


def load_seen(parent_dir):
    seen = set()
    for fname in (CLEANED_CSV, DISCARD_CSV):
        fpath = os.path.join(parent_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, newline="") as f:
                for row in csv.DictReader(f):
                    seen.add(row["image_path"])
    return seen


def append_row(parent_dir, csv_name, image_path, mask_path):
    fpath    = os.path.join(parent_dir, csv_name)
    new_file = not os.path.exists(fpath)
    with open(fpath, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if new_file:
            w.writeheader()
        w.writerow({"image_path": image_path, "mask_path": mask_path})


def load_pairs(parent_dir):
    images_dir = os.path.join(parent_dir, "Images")
    masks_dir  = os.path.join(parent_dir, "Mask")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    pairs = []
    for fname in sorted(os.listdir(images_dir)):
        if os.path.splitext(fname)[1].lower() not in exts:
            continue
        img_path  = os.path.join(images_dir, fname)
        mask_name = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(masks_dir, mask_name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(masks_dir, fname)
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    return pairs


def load_pairs_from_keep_csv(parent_dir):
    """Load image/mask pairs from keep.csv, remapping paths to parent_dir."""
    keep_path  = os.path.join(parent_dir, KEEP_CSV)
    images_dir = os.path.join(parent_dir, "Images")
    masks_dir  = os.path.join(parent_dir, "Mask")
    pairs = []
    with open(keep_path, newline="") as f:
        for row in csv.DictReader(f):
            fname     = os.path.basename(row["image_path"])
            img_path  = os.path.join(images_dir, fname)
            mask_name = os.path.splitext(fname)[0] + ".png"
            mask_path = os.path.join(masks_dir, mask_name)
            if os.path.exists(img_path) and os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
    return pairs


def load_mask_from_file(mask_path, display_size=None):
    """Load mask PNG as uint8 {0,1} numpy array.

    If display_size (W, H) is given and differs from the mask's size, the mask
    is downscaled to display_size using nearest-neighbour so edits stay in
    display-pixel space.  The original file is not modified.
    """
    mask_np = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    mask_bin = (mask_np > 127).astype(np.uint8)
    if display_size is not None:
        dW, dH = display_size
        if (mask_bin.shape[1], mask_bin.shape[0]) != (dW, dH):
            mask_pil = Image.fromarray((mask_bin * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((dW, dH), Image.NEAREST)
            mask_bin = (np.asarray(mask_pil) > 127).astype(np.uint8)
    return mask_bin


def mask_to_rgba_png_bytes(mask_u8):
    """Convert uint8 {0,1} mask → RGBA PNG bytes with red overlay."""
    H, W  = mask_u8.shape
    rgba  = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[mask_u8 > 0] = [255, 50, 50, 140]
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    buf.seek(0)
    return buf


def load_current(s):
    """Load mask for the current queue index into STATE."""
    img_path, mask_path = s["queue"][s["idx"]]
    with Image.open(img_path) as im:
        orig_W, orig_H = im.size
    dW, dH = compute_display_size(orig_W, orig_H)
    s["orig_size"]         = (orig_W, orig_H)
    s["display_size"]      = (dW, dH)
    s["current_mask"]      = load_mask_from_file(mask_path, display_size=(dW, dH))
    s["current_mask_path"] = mask_path
    s["ref_radius"]        = compute_ref_radius(img_path, (dW, dH))


# ── HTML ───────────────────────────────────────────────────────────────────────

HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Follicle Mask Reviewer</title>
<style>
  * { box-sizing: border-box; }
  body   { font-family: sans-serif; background: #1a1a1a; color: #eee;
           text-align: center; margin: 0; padding: 20px; user-select: none; }
  #info  { font-size: .9em; color: #aaa; margin-bottom: 4px; }
  #fname { font-size: 1em;  color: #ccc; margin-bottom: 8px; word-break: break-all; }
  #status { margin-bottom: 10px; font-size: .9em;
            display: flex; gap: 20px; justify-content: center; align-items: center; }
  .badge        { padding: 2px 10px; border-radius: 4px; font-weight: bold; }
  .badge-delete { background: #c0392b; color: #fff; }
  .badge-add    { background: #27ae60; color: #fff; }
  .badge-on     { background: #2980b9; color: #fff; }
  .badge-off    { background: #555;    color: #aaa; }
  .badge-star   { background: #f39c12; color: #fff; }
  .badge-unstar { background: #555;    color: #aaa; }
  canvas { display: block; margin: 0 auto;
           max-width: 700px; max-height: 700px;
           border: 2px solid #444; cursor: crosshair; }
  #btns  { margin-top: 14px; display: flex; gap: 30px; justify-content: center; }
  button { padding: 10px 36px; font-size: 1em; font-weight: bold;
           border: none; border-radius: 6px; cursor: pointer; }
  #bkeep { background: #4CAF50; color: #fff; }
  #bdisc { background: #f44336; color: #fff; }
  button:hover { opacity: .85; }
  #keys  { font-size: .75em; color: #666; margin-top: 12px; }
  #done  { font-size: 1.4em; margin-top: 40px; color: #4CAF50; display: none; }
</style>
</head>
<body>
<div id="info">Loading…</div>
<div id="fname"></div>
<div id="status">
  <span>Mode: <span id="mode-badge" class="badge badge-delete">DELETE</span></span>
  <span>Mask: <span id="mask-badge" class="badge badge-on">ON</span></span>
  <span><span id="star-badge" class="badge badge-unstar">☆ STAR</span></span>
</div>
<canvas id="c"></canvas>
<div id="btns">
  <button id="bdisc" onclick="decide('discard')">Discard (d)</button>
  <button id="bkeep" onclick="decide('keep')">Keep (k)</button>
</div>
<div id="keys">
  g: toggle mask &nbsp;|&nbsp; e: delete mode &nbsp;|&nbsp; a: add mode
  &nbsp;|&nbsp; k: keep &nbsp;|&nbsp; d: discard &nbsp;|&nbsp; *: star
</div>
<div id="done"></div>

<script>
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');

let baseImg        = null;
let maskImg        = null;
let showMask       = true;
let editMode       = 'delete';
let isStarred      = false;
let dragStart      = null;   // add-mode: {x,y} in image coords
let deleteBoxAnchor = null;  // delete-mode drag: {clientX, clientY, imgPos}
let pending        = false;
let refRadius      = 0;
const DRAG_THRESHOLD = 5;    // display pixels before a click becomes a box-drag

// ── Utilities ──────────────────────────────────────────────────────────────

function loadImg(src) {
  return new Promise((res, rej) => {
    const i = new Image();
    i.onload  = () => res(i);
    i.onerror = rej;
    i.src     = src;
  });
}

function clientToImage(e) {
  const rect   = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: Math.round((e.clientX - rect.left) * scaleX),
    y: Math.round((e.clientY - rect.top)  * scaleY),
  };
}

function redraw(circlePreview, boxPreview) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (baseImg) ctx.drawImage(baseImg, 0, 0);
  if (showMask && maskImg) ctx.drawImage(maskImg, 0, 0);
  if (circlePreview) {
    const { x, y, r } = circlePreview;
    ctx.beginPath();
    ctx.arc(x, y, Math.max(1, r), 0, Math.PI * 2);
    ctx.fillStyle   = 'rgba(255, 255, 0, 0.35)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 220, 0, 0.9)';
    ctx.lineWidth   = 2;
    ctx.stroke();
  }
  if (boxPreview) {
    const { x0, y0, x1, y1 } = boxPreview;
    ctx.fillStyle   = 'rgba(255, 80, 80, 0.15)';
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    ctx.strokeStyle = 'rgba(255, 80, 80, 0.9)';
    ctx.lineWidth   = 2;
    ctx.setLineDash([6, 3]);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.setLineDash([]);
  }
  if (refRadius > 0) {
    const pad = 10;
    const cx  = canvas.width  - refRadius - pad;
    const cy  = canvas.height - refRadius - pad;
    ctx.beginPath();
    ctx.arc(cx, cy, refRadius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.fill();
    ctx.font         = '11px sans-serif';
    ctx.fillStyle    = 'rgba(255, 255, 255, 0.85)';
    ctx.textAlign    = 'center';
    ctx.fillText('min follicle', cx, cy - refRadius - 4);
  }
}

function updateModeUI() {
  const badge = document.getElementById('mode-badge');
  if (editMode === 'delete') {
    badge.textContent = 'DELETE';
    badge.className   = 'badge badge-delete';
    canvas.style.cursor = 'crosshair';
  } else {
    badge.textContent = 'ADD';
    badge.className   = 'badge badge-add';
    canvas.style.cursor = 'cell';
  }
}

function updateMaskUI() {
  const badge = document.getElementById('mask-badge');
  badge.textContent = showMask ? 'ON' : 'OFF';
  badge.className   = showMask ? 'badge badge-on' : 'badge badge-off';
}

function updateStarUI() {
  const badge = document.getElementById('star-badge');
  badge.textContent = isStarred ? '★ STARRED' : '☆ STAR';
  badge.className   = isStarred ? 'badge badge-star' : 'badge badge-unstar';
}

// ── Data fetching ───────────────────────────────────────────────────────────

async function fetchMask() {
  const resp = await fetch('/mask_png?t=' + Date.now());
  if (!resp.ok) return null;
  return loadImg(URL.createObjectURL(await resp.blob()));
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
  document.getElementById('info').textContent  =
    d.reviewed + ' / ' + d.total + ' reviewed  |  ' + d.remaining + ' remaining';
  document.getElementById('fname').textContent = d.filename;
  refRadius = d.ref_radius || 0;
  isStarred = d.is_starred || false;
  updateStarUI();

  const [img, msk] = await Promise.all([
    loadImg('/file?p=' + encodeURIComponent(d.image_path)),
    fetchMask(),
  ]);
  baseImg        = img;
  maskImg        = msk;
  canvas.width   = img.naturalWidth;
  canvas.height  = img.naturalHeight;
  redraw();
}

async function applyEdit(body) {
  if (pending) return;
  pending = true;
  try {
    const resp = await fetch('/mask_edit', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    if (!resp.ok) return;
    maskImg = await loadImg(URL.createObjectURL(await resp.blob()));
    redraw();
  } finally {
    pending = false;
  }
}

async function decide(action) {
  dragStart = null;
  deleteBoxAnchor = null;
  await fetch('/decide', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ action }),
  });
  load();
}

// ── Canvas mouse events ─────────────────────────────────────────────────────

canvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  const pos = clientToImage(e);
  if (editMode === 'add') {
    dragStart = pos;
  } else {
    deleteBoxAnchor = { clientX: e.clientX, clientY: e.clientY, imgPos: pos };
  }
});

canvas.addEventListener('mousemove', e => {
  if (editMode === 'add' && dragStart) {
    const pos = clientToImage(e);
    const r   = Math.round(Math.hypot(pos.x - dragStart.x, pos.y - dragStart.y));
    redraw({ x: dragStart.x, y: dragStart.y, r }, null);
  } else if (editMode === 'delete' && deleteBoxAnchor) {
    const dx = e.clientX - deleteBoxAnchor.clientX;
    const dy = e.clientY - deleteBoxAnchor.clientY;
    if (Math.hypot(dx, dy) > DRAG_THRESHOLD) {
      const pos = clientToImage(e);
      const a   = deleteBoxAnchor.imgPos;
      redraw(null, {
        x0: Math.min(a.x, pos.x), y0: Math.min(a.y, pos.y),
        x1: Math.max(a.x, pos.x), y1: Math.max(a.y, pos.y),
      });
    }
  }
});

canvas.addEventListener('mouseup', e => {
  if (editMode === 'add' && dragStart) {
    const pos = clientToImage(e);
    const r   = Math.round(Math.hypot(pos.x - dragStart.x, pos.y - dragStart.y));
    const cx  = dragStart.x;
    const cy  = dragStart.y;
    dragStart = null;
    if (r > 0) {
      applyEdit({ action: 'add', x: cx, y: cy, r });
    } else {
      redraw();
    }
  } else if (editMode === 'delete' && deleteBoxAnchor) {
    const dx     = e.clientX - deleteBoxAnchor.clientX;
    const dy     = e.clientY - deleteBoxAnchor.clientY;
    const anchor = deleteBoxAnchor;
    deleteBoxAnchor = null;
    if (Math.hypot(dx, dy) > DRAG_THRESHOLD) {
      const pos = clientToImage(e);
      const a   = anchor.imgPos;
      applyEdit({
        action: 'delete_box',
        x0: Math.min(a.x, pos.x), y0: Math.min(a.y, pos.y),
        x1: Math.max(a.x, pos.x), y1: Math.max(a.y, pos.y),
      });
    } else {
      applyEdit({ action: 'delete', x: anchor.imgPos.x, y: anchor.imgPos.y });
    }
  }
});

canvas.addEventListener('mouseleave', () => {
  if (dragStart)       { dragStart = null;       redraw(); }
  if (deleteBoxAnchor) { deleteBoxAnchor = null;  redraw(); }
});

// ── Keyboard events ─────────────────────────────────────────────────────────

document.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
  switch (e.key) {
    case 'g':
      showMask = !showMask;
      updateMaskUI();
      redraw();
      break;
    case 'e':
      editMode = 'delete';
      updateModeUI();
      break;
    case 'a':
      editMode = 'add';
      updateModeUI();
      break;
    case 'k': decide('keep');    break;
    case 'd': decide('discard'); break;
    case '*':
      fetch('/star', { method: 'POST' })
        .then(r => r.json())
        .then(d => { isStarred = d.starred; updateStarUI(); });
      break;
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
    s     = STATE
    idx   = s["idx"]
    queue = s["queue"]
    if idx >= len(queue):
        return jsonify({"done": True, "message": f"All {s['total']} pairs reviewed!"})
    img_path, mask_path = queue[idx]
    return jsonify({
        "done":       False,
        "filename":   os.path.basename(img_path),
        "image_path": img_path,
        "mask_path":  mask_path,
        "reviewed":   s["seen_count"] + idx,
        "remaining":  len(queue) - idx,
        "total":      s["total"],
        "ref_radius": s.get("ref_radius", 0),
        "is_starred": img_path in s.get("starred", set()),
    })


@app.route("/file")
def serve_file():
    path = request.args.get("p", "")
    if not os.path.isfile(path):
        return "not found", 404
    img = Image.open(path).convert("RGB")
    dW, dH = compute_display_size(*img.size)
    if (dW, dH) != img.size:
        img = img.resize((dW, dH), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    return send_file(path)


@app.route("/mask_png")
def mask_png():
    mask = STATE.get("current_mask")
    if mask is None:
        return "no mask", 404
    return send_file(mask_to_rgba_png_bytes(mask), mimetype="image/png")


@app.route("/mask_edit", methods=["POST"])
def mask_edit():
    data   = request.get_json()
    action = data["action"]
    s      = STATE
    mask   = s.get("current_mask")
    if mask is None:
        return "no mask", 404

    mask = mask.copy()
    H, W = mask.shape

    if action == "delete":
        x, y = int(data["x"]), int(data["y"])
        if 0 <= y < H and 0 <= x < W and mask[y, x] > 0:
            _, labels = cv.connectedComponents(mask)
            label = int(labels[y, x])
            if label > 0:
                mask[labels == label] = 0

    elif action == "delete_box":
        x0 = max(0,     int(min(data["x0"], data["x1"])))
        y0 = max(0,     int(min(data["y0"], data["y1"])))
        x1 = min(W - 1, int(max(data["x0"], data["x1"])))
        y1 = min(H - 1, int(max(data["y0"], data["y1"])))
        _, labels = cv.connectedComponents(mask)
        box_labels = set(np.unique(labels[y0:y1 + 1, x0:x1 + 1])) - {0}
        for lbl in box_labels:
            mask[labels == lbl] = 0

    elif action == "add":
        cx = int(data["x"])
        cy = int(data["y"])
        r  = max(1, int(data.get("r", 10)))
        cv.circle(mask, (cx, cy), r, 1, -1)

    s["current_mask"] = mask
    return send_file(mask_to_rgba_png_bytes(mask), mimetype="image/png")


@app.route("/decide", methods=["POST"])
def decide():
    action = request.get_json().get("action")
    s      = STATE
    idx    = s["idx"]
    queue  = s["queue"]
    if idx >= len(queue):
        return jsonify({"status": "done"})

    img_path, orig_mask_path = queue[idx]

    if action == "keep":
        mask = s.get("current_mask")
        if mask is not None:
            orig_W, orig_H = s.get("orig_size", (mask.shape[1], mask.shape[0]))
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            if mask_pil.size != (orig_W, orig_H):
                mask_pil = mask_pil.resize((orig_W, orig_H), Image.NEAREST)
            mask_pil.save(orig_mask_path)
        append_row(s["parent_dir"], CLEANED_CSV, img_path, orig_mask_path)
    else:
        append_row(s["parent_dir"], DISCARD_CSV, img_path, orig_mask_path)

    s["idx"] += 1
    if s["idx"] < len(queue):
        load_current(s)

    return jsonify({"status": "ok"})


@app.route("/star", methods=["POST"])
def star():
    s   = STATE
    idx = s["idx"]
    if idx >= len(s["queue"]):
        return jsonify({"starred": False})
    img_path, mask_path = s["queue"][idx]
    if img_path not in s["starred"]:
        append_row(s["parent_dir"], STARRED_CSV, img_path, mask_path)
        s["starred"].add(img_path)
    return jsonify({"starred": img_path in s["starred"]})


# ── Entry point ────────────────────────────────────────────────────────────────

METADATA_CSV = "/home/Trachoma/data/all_metadata.csv"
EXCLUDE_SOURCES = {"SOCIT", "SOCIT_R"}


def build_source_map():
    """Return stem → source dict from all_metadata.csv."""
    stem_to_source = {}
    with open(METADATA_CSV, newline="") as f:
        for row in csv.DictReader(f):
            stem = os.path.splitext(os.path.basename(row["image_path"]))[0]
            stem_to_source[stem] = row["source"]
    return stem_to_source


def interleave_by_source(queue, exclude_sources=EXCLUDE_SOURCES):
    """
    Group pairs by dataset source and interleave round-robin so every
    review session sees a uniform mix across datasets.  Any excluded
    sources (SOCIT) are placed at the end if they exist at all.
    """
    stem_to_source = build_source_map()

    buckets = {}   # source → [pairs]
    excluded = []
    for pair in queue:
        stem = os.path.splitext(os.path.basename(pair[0]))[0]
        src  = stem_to_source.get(stem, "UNKNOWN")
        if src in exclude_sources:
            excluded.append(pair)
        else:
            buckets.setdefault(src, []).append(pair)

    # Sort bucket keys for determinism; print summary
    sorted_sources = sorted(buckets.keys())
    for src in sorted_sources:
        print(f"  {src}: {len(buckets[src])}")
    if excluded:
        print(f"  (excluded/SOCIT appended at end): {len(excluded)}")

    # Round-robin interleave
    interleaved = [
        pair
        for pair in itertools.chain.from_iterable(
            itertools.zip_longest(*[buckets[s] for s in sorted_sources])
        )
        if pair is not None
    ]
    return interleaved + excluded


def main():
    parent_dir = (
        sys.argv[1] if len(sys.argv) > 1
        else os.path.join(SCRIPT_DIR, "manual_check")
    )
    if not os.path.isdir(parent_dir):
        print(f"Error: directory not found: {parent_dir}")
        sys.exit(1)

    pairs = load_pairs_from_keep_csv(parent_dir)
    seen  = load_seen(parent_dir)
    queue = [p for p in pairs if p[0] not in seen]
    queue = interleave_by_source(queue)

    STATE.update({
        "parent_dir":        parent_dir,
        "queue":             queue,
        "total":             len(pairs),
        "seen_count":        len(pairs) - len(queue),
        "idx":               0,
        "current_mask":      None,
        "current_mask_path": None,
        "starred":           load_starred(parent_dir),
    })

    if queue:
        load_current(STATE)

    print(f"Total: {len(pairs)}  |  Seen: {STATE['seen_count']}  |  Remaining: {len(queue)}")
    print("Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
