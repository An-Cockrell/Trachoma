#!/usr/bin/env python3
"""
Annotation tool for drawing follicle masks from scratch.

Run from any directory. Open http://localhost:5000

Creates ./annotations/images/, ./annotations/masks/, keep.csv, and
discard.csv in the current working directory on startup.

Controls:
  a         — add mode (default): click+drag to draw a follicle circle
  e         — delete mode: click a blob to erase it; drag a box to erase all inside
  g         — toggle mask overlay on/off
  k         — keep  (saves image + drawn mask to annotations/)
  d         — discard
"""

import os
import io
import csv
import shutil

import cv2 as cv
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, send_file

CWD             = os.getcwd()
ANNOTATIONS_DIR = os.path.join(CWD, "annotations")
IMAGES_OUT_DIR  = os.path.join(ANNOTATIONS_DIR, "images")
MASKS_OUT_DIR   = os.path.join(ANNOTATIONS_DIR, "masks")
KEEP_CSV        = "keep.csv"
DISCARD_CSV     = "discard.csv"
CSV_HEADER      = ["image_path", "mask_path"]

app   = Flask(__name__)
STATE = {"ready": False}


# ── Setup ──────────────────────────────────────────────────────────────────────

def ensure_output_dirs():
    os.makedirs(IMAGES_OUT_DIR, exist_ok=True)
    os.makedirs(MASKS_OUT_DIR,  exist_ok=True)


# ── CSV helpers ────────────────────────────────────────────────────────────────

def load_seen():
    seen = set()
    for fname in (KEEP_CSV, DISCARD_CSV):
        fpath = os.path.join(ANNOTATIONS_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, newline="") as f:
                for row in csv.DictReader(f):
                    seen.add(row["image_path"])
    return seen


def append_row(csv_name, image_path, mask_path):
    fpath    = os.path.join(ANNOTATIONS_DIR, csv_name)
    new_file = not os.path.exists(fpath)
    with open(fpath, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if new_file:
            w.writeheader()
        w.writerow({"image_path": image_path, "mask_path": mask_path or ""})


# ── Image helpers ──────────────────────────────────────────────────────────────

def load_images_from_folder(image_folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    images = []
    for fname in sorted(os.listdir(image_folder)):
        if os.path.splitext(fname)[1].lower() not in exts:
            continue
        images.append(os.path.normpath(os.path.join(image_folder, fname)))
    return images


def load_images_from_txt(txt_path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    images = []
    with open(txt_path) as f:
        for line in f:
            p = line.strip()
            if not p or p.startswith("#"):
                continue
            if os.path.splitext(p)[1].lower() not in exts:
                continue
            if os.path.isfile(p):
                images.append(p)
    return images


def load_current(s):
    while s["idx"] < len(s["queue"]):
        img_path = s["queue"][s["idx"]]
        try:
            with Image.open(img_path) as im:
                W, H = im.size
            s["current_mask"]  = np.zeros((H, W), dtype=np.uint8)
            s["current_image"] = img_path
            return
        except Exception:
            s["idx"] += 1


def mask_to_rgba_png_bytes(mask_u8):
    H, W = mask_u8.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[mask_u8 > 0] = [255, 50, 50, 140]
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
<title>Annotation Tool</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: sans-serif; background: #1a1a1a; color: #eee;
         text-align: center; margin: 0; padding: 20px; user-select: none; }

  /* ── Loader ── */
  #loader { max-width: 520px; margin: 80px auto; }
  #loader h2 { font-size: 1.8em; margin-bottom: 6px; }
  #loader p  { color: #aaa; margin-bottom: 20px; }
  #folder-path { font-size: .85em; color: #888; background: #252525;
                 padding: 8px 14px; border-radius: 4px; margin-bottom: 6px;
                 word-break: break-all; min-height: 34px; }
  #manual-entry { display: none; margin: 10px 0; }
  #manual-entry input { width: 100%; padding: 8px 10px; background: #252525;
                        color: #eee; border: 1px solid #444; border-radius: 4px;
                        font-size: .9em; margin-bottom: 6px; }
  #load-error  { color: #f44336; margin: 8px 0; font-size: .9em; min-height: 1.2em; }
  #folder-stats { color: #4CAF50; margin-top: 10px; font-size: .9em; }

  /* ── Annotator ── */
  #change-folder { font-size: .75em; color: #555; cursor: pointer;
                   text-decoration: underline; margin-bottom: 8px;
                   display: inline-block; }
  #change-folder:hover { color: #888; }
  #info  { font-size: .9em; color: #aaa; margin-bottom: 4px; }
  #fname { font-size: 1em;  color: #ccc; margin-bottom: 8px; word-break: break-all; }
  #status { margin-bottom: 10px; font-size: .9em;
            display: flex; gap: 20px; justify-content: center; align-items: center; }
  .badge        { padding: 2px 10px; border-radius: 4px; font-weight: bold; }
  .badge-delete { background: #c0392b; color: #fff; }
  .badge-add    { background: #27ae60; color: #fff; }
  .badge-on     { background: #2980b9; color: #fff; }
  .badge-off    { background: #555;    color: #aaa; }
  canvas { display: block; margin: 0 auto;
           max-width: 700px; max-height: 700px;
           border: 2px solid #444; cursor: cell; }
  #btns  { margin-top: 14px; display: flex; gap: 30px; justify-content: center; }
  #keys  { font-size: .75em; color: #666; margin-top: 12px; }
  #done  { font-size: 1.4em; margin-top: 40px; color: #4CAF50; display: none; }
  #done-sub { font-size: .75em; color: #666; margin-top: 8px; }

  /* ── Shared buttons ── */
  button { padding: 10px 36px; font-size: 1em; font-weight: bold;
           border: none; border-radius: 6px; cursor: pointer; }
  button:hover { opacity: .85; }
  #bkeep     { background: #4CAF50; color: #fff; }
  #bdisc     { background: #f44336; color: #fff; }
  .btn-blue  { background: #2980b9; color: #fff; }
  .btn-grey  { background: #444;    color: #ccc; }
</style>
</head>
<body>

<!-- ════════════════ LOADER ════════════════ -->
<div id="loader">
  <h2>Annotation Tool</h2>
  <p>Select a folder of images to begin annotating follicles.</p>

  <div id="folder-path">No folder selected</div>
  <div id="load-error"></div>

  <div style="display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin-bottom:10px;">
    <button class="btn-blue" onclick="browseFolder()">Load from folder</button>
    <button class="btn-blue" onclick="browseTxtFile()">Load from .txt file</button>
    <button class="btn-grey" onclick="toggleManual()">Enter path manually</button>
  </div>

  <div id="manual-entry">
    <input id="manual-path" type="text" placeholder="/path/to/folder  or  /path/to/list.txt"
           onkeydown="if(event.key==='Enter') loadManual(document.getElementById('manual-path').value)">
    <button class="btn-blue"
            onclick="loadManual(document.getElementById('manual-path').value)">Go</button>
  </div>

  <div id="folder-stats"></div>
</div>

<!-- ════════════════ ANNOTATOR ════════════════ -->
<div id="annotator" style="display:none">
  <span id="change-folder" onclick="showLoader()">&#9650; Change Folder</span>
  <div id="info">Loading&hellip;</div>
  <div id="fname"></div>
  <div id="status">
    <span>Mode: <span id="mode-badge" class="badge badge-add">ADD</span></span>
    <span>Mask: <span id="mask-badge" class="badge badge-on">ON</span></span>
  </div>
  <canvas id="c"></canvas>
  <div id="btns">
    <button id="bdisc" onclick="decide('discard')">Discard (d)</button>
    <button id="bkeep" onclick="decide('keep')">Keep (k)</button>
  </div>
  <div id="keys">
    a: add follicle &nbsp;|&nbsp; e: delete mode &nbsp;|&nbsp; g: toggle mask
    &nbsp;|&nbsp; k: keep &nbsp;|&nbsp; d: discard
  </div>
  <div id="done">
    <div id="done-msg"></div>
    <div id="done-sub">
      <span style="cursor:pointer;text-decoration:underline;color:#888"
            onclick="showLoader()">Load a different folder</span>
    </div>
  </div>
</div>

<script>
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');

let baseImg         = null;
let maskImg         = null;
let showMask        = true;
let editMode        = 'add';   // default: draw follicles
let dragStart       = null;
let deleteBoxAnchor = null;
let pending         = false;
const DRAG_THRESHOLD = 5;

// ── Loader ───────────────────────────────────────────────────────────────────

function setError(msg) {
  document.getElementById('load-error').textContent = msg || '';
}

function toggleManual() {
  const el = document.getElementById('manual-entry');
  el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

async function browseFolder() {
  setError('');
  const res = await fetch('/browse').then(r => r.json());
  if (!res.folder) {
    if (res.error) {
      setError('Native browser unavailable: ' + res.error);
      document.getElementById('manual-entry').style.display = 'block';
    }
    return;
  }
  await loadFolder(res.folder);
}

async function browseTxtFile() {
  setError('');
  const res = await fetch('/browse_file').then(r => r.json());
  if (!res.path) {
    if (res.error) {
      setError('Native browser unavailable: ' + res.error);
      document.getElementById('manual-entry').style.display = 'block';
    }
    return;
  }
  await loadTxt(res.path);
}

function loadManual(val) {
  if (!val || !val.trim()) { setError('No path specified.'); return; }
  if (val.trim().toLowerCase().endsWith('.txt')) {
    loadTxt(val.trim());
  } else {
    loadFolder(val.trim());
  }
}

async function loadFolder(folder) {
  if (!folder || !folder.trim()) { setError('No folder specified.'); return; }
  setError('');
  document.getElementById('folder-path').textContent = folder;

  const res = await fetch('/load_folder', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ folder: folder.trim() }),
  }).then(r => r.json());

  if (res.error) { setError(res.error); return; }
  startAnnotating(folder, res);
}

async function loadTxt(path) {
  if (!path || !path.trim()) { setError('No file specified.'); return; }
  setError('');
  document.getElementById('folder-path').textContent = path;

  const res = await fetch('/load_txt', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ path: path.trim() }),
  }).then(r => r.json());

  if (res.error) { setError(res.error); return; }
  startAnnotating(path, res);
}

function startAnnotating(label, res) {
  document.getElementById('folder-stats').textContent =
    res.total + ' images found  |  ' +
    res.seen  + ' already reviewed  |  ' +
    res.remaining + ' remaining';

  document.getElementById('loader').style.display    = 'none';
  document.getElementById('annotator').style.display = 'block';
  document.getElementById('done').style.display      = 'none';
  canvas.style.display = 'block';
  document.getElementById('btns').style.display = 'flex';
  load();
}

function showLoader() {
  document.getElementById('annotator').style.display = 'none';
  document.getElementById('loader').style.display    = 'block';
  document.getElementById('folder-stats').textContent = '';
  setError('');
}

// ── Utilities ────────────────────────────────────────────────────────────────

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
}

function updateModeUI() {
  const badge = document.getElementById('mode-badge');
  if (editMode === 'delete') {
    badge.textContent   = 'DELETE';
    badge.className     = 'badge badge-delete';
    canvas.style.cursor = 'crosshair';
  } else {
    badge.textContent   = 'ADD';
    badge.className     = 'badge badge-add';
    canvas.style.cursor = 'cell';
  }
}

function updateMaskUI() {
  const badge = document.getElementById('mask-badge');
  badge.textContent = showMask ? 'ON' : 'OFF';
  badge.className   = showMask ? 'badge badge-on' : 'badge badge-off';
}

// ── Data fetching ─────────────────────────────────────────────────────────────

async function fetchMask() {
  const resp = await fetch('/mask_png?t=' + Date.now());
  if (!resp.ok) return null;
  return loadImg(URL.createObjectURL(await resp.blob()));
}

async function load() {
  const d = await fetch('/current').then(r => r.json());
  if (d.done) {
    canvas.style.display = 'none';
    document.getElementById('btns').style.display  = 'none';
    document.getElementById('done').style.display  = 'block';
    document.getElementById('done-msg').textContent = d.message;
    document.getElementById('info').textContent    = '';
    return;
  }

  document.getElementById('info').textContent  =
    d.reviewed + ' / ' + d.total + ' reviewed  |  ' + d.remaining + ' remaining';
  document.getElementById('fname').textContent = d.filename;

  // Reset to add mode for each new image
  editMode = 'add';
  updateModeUI();
  updateMaskUI();

  const [img, msk] = await Promise.all([
    loadImg('/file?p=' + encodeURIComponent(d.image_path)),
    fetchMask(),
  ]);
  baseImg       = img;
  maskImg       = msk;
  canvas.width  = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.style.display = 'block';
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
  dragStart       = null;
  deleteBoxAnchor = null;
  await fetch('/decide', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ action }),
  });
  load();
}

// ── Canvas mouse events ───────────────────────────────────────────────────────

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

// ── Keyboard events ───────────────────────────────────────────────────────────

document.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
  if (document.getElementById('annotator').style.display === 'none') return;
  switch (e.key) {
    case 'g':
      showMask = !showMask;
      updateMaskUI();
      redraw();
      break;
    case 'e': editMode = 'delete'; updateModeUI(); break;
    case 'a': editMode = 'add';    updateModeUI(); break;
    case 'k': decide('keep');    break;
    case 'd': decide('discard'); break;
  }
});
</script>
</body>
</html>
"""


# ── Flask routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML


@app.route("/browse")
def browse():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select Image Folder")
        root.destroy()
        return jsonify({"folder": folder or "", "error": None})
    except Exception as exc:
        return jsonify({"folder": "", "error": str(exc)})


@app.route("/browse_file")
def browse_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select image-list .txt file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        root.destroy()
        return jsonify({"path": path or "", "error": None})
    except Exception as exc:
        return jsonify({"path": "", "error": str(exc)})


@app.route("/load_txt", methods=["POST"])
def load_txt():
    txt_path = request.get_json().get("path", "").strip()
    if not os.path.isfile(txt_path):
        return jsonify({"error": f"File not found: {txt_path}"}), 400
    if not txt_path.lower().endswith(".txt"):
        return jsonify({"error": "Expected a .txt file."}), 400

    images = load_images_from_txt(txt_path)
    if not images:
        return jsonify({"error": "No valid image paths found in that file."}), 400

    seen  = load_seen()
    queue = [p for p in images if p not in seen]

    STATE.update({
        "ready":         True,
        "image_folder":  os.path.dirname(txt_path),
        "queue":         queue,
        "total":         len(images),
        "seen_count":    len(images) - len(queue),
        "idx":           0,
        "current_mask":  None,
        "current_image": None,
    })

    if queue:
        load_current(STATE)

    return jsonify({
        "total":     len(images),
        "seen":      len(images) - len(queue),
        "remaining": len(queue),
    })


@app.route("/load_folder", methods=["POST"])
def load_folder():
    folder = os.path.normpath(request.get_json().get("folder", "").strip())
    if not os.path.isdir(folder):
        return jsonify({"error": f"Not a valid directory: {folder}"}), 400

    images = load_images_from_folder(folder)
    if not images:
        return jsonify({"error": "No images found in that folder."}), 400

    seen  = load_seen()
    queue = [p for p in images if p not in seen]

    STATE.update({
        "ready":         True,
        "image_folder":  folder,
        "queue":         queue,
        "total":         len(images),
        "seen_count":    len(images) - len(queue),
        "idx":           0,
        "current_mask":  None,
        "current_image": None,
    })

    if queue:
        load_current(STATE)

    return jsonify({
        "total":     len(images),
        "seen":      len(images) - len(queue),
        "remaining": len(queue),
    })


@app.route("/current")
def current():
    s = STATE
    if not s.get("ready"):
        return jsonify({"done": False, "not_loaded": True})
    idx   = s["idx"]
    queue = s["queue"]
    if idx >= len(queue):
        return jsonify({"done": True, "message": f"All {s['total']} images reviewed!"})
    img_path = queue[idx]
    return jsonify({
        "done":       False,
        "filename":   os.path.basename(img_path),
        "image_path": img_path,
        "reviewed":   s["seen_count"] + idx,
        "remaining":  len(queue) - idx,
        "total":      s["total"],
    })


@app.route("/file")
def serve_file():
    path = request.args.get("p", "")
    if not os.path.isfile(path):
        return "not found", 404
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

    img_path = queue[idx]

    if action == "keep":
        img_out = os.path.join(IMAGES_OUT_DIR, os.path.basename(img_path))
        shutil.copy2(img_path, img_out)

        mask      = s.get("current_mask")
        mask_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        mask_out  = os.path.join(MASKS_OUT_DIR, mask_name)
        if mask is not None:
            Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(mask_out)
        else:
            Image.new("L", (1, 1), 0).save(mask_out)   # should never happen

        append_row(KEEP_CSV, img_path, mask_out)
    else:
        append_row(DISCARD_CSV, img_path, "")

    s["idx"] += 1
    if s["idx"] < len(queue):
        load_current(s)

    return jsonify({"status": "ok"})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensure_output_dirs()
    print(f"Annotations output: {ANNOTATIONS_DIR}")
    print("Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
