#!/usr/bin/env python3
"""
Zero-follicle review app.

Loads all TF-negative images from all_metadata.csv and presents each one
for quick labelling. Shuffled order is generated once and stored so sessions
are resumable.

Output : MultipleFollicleImages/zero_follicles/keep.csv    (zero follicles confirmed)
         MultipleFollicleImages/zero_follicles/discard.csv  (discarded / unsure)
Hotkeys: k = Zero Follicles  |  d = Discard

Open http://localhost:5000
"""

import csv
import os
import random

from flask import Flask, jsonify, request, send_file

app   = Flask(__name__)
STATE = {}

_HERE        = os.path.dirname(os.path.abspath(__file__))
METADATA_CSV = "/home/Trachoma/data/all_metadata.csv"
OUT_DIR      = os.path.join(_HERE, "MultipleFollicleImages", "zero_follicles")
SHUFFLED_FILE = os.path.join(OUT_DIR, "shuffled_list.txt")
KEEP_CSV     = os.path.join(OUT_DIR, "keep.csv")
DISCARD_CSV  = os.path.join(OUT_DIR, "discard.csv")
CSV_HEADER   = ["image_path"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_tf_negatives():
    paths = []
    with open(METADATA_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] == "0":
                paths.append(row["image_path"])
    return paths


def get_or_create_shuffled_list():
    if os.path.exists(SHUFFLED_FILE):
        with open(SHUFFLED_FILE) as f:
            items = [line.strip() for line in f if line.strip()]
        print(f"Loaded existing shuffled list: {len(items)} images")
        return items

    negatives = load_tf_negatives()
    random.shuffle(negatives)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SHUFFLED_FILE, "w") as f:
        for p in negatives:
            f.write(p + "\n")

    print(f"Generated shuffled list: {len(negatives)} TF-negative images")
    return negatives


def load_seen_set():
    seen = set()
    for fpath in (KEEP_CSV, DISCARD_CSV):
        if os.path.exists(fpath):
            with open(fpath, newline="") as f:
                for row in csv.DictReader(f):
                    seen.add(row["image_path"])
    return seen


def append_row(csv_path, image_path):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if new_file:
            w.writeheader()
        w.writerow({"image_path": image_path})


# ── HTML ───────────────────────────────────────────────────────────────────────

HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Zero Follicle Review</title>
<style>
  * { box-sizing: border-box; }
  body   { font-family: sans-serif; background: #1a1a1a; color: #eee;
           text-align: center; margin: 0; padding: 20px; user-select: none; }
  #info  { font-size: .9em; color: #aaa; margin-bottom: 4px; }
  #fname { font-size: 1em; color: #ccc; margin-bottom: 14px; word-break: break-all; }
  img#photo { display: block; margin: 0 auto;
              max-width: 900px; max-height: 700px;
              border: 2px solid #444; }
  #btns { margin-top: 18px; display: flex; gap: 40px; justify-content: center; }
  .btn-discard { padding: 12px 52px; font-size: 1.1em; font-weight: bold;
                 border: none; border-radius: 6px; cursor: pointer;
                 background: #c0392b; color: #fff; }
  .btn-zero   { padding: 12px 52px; font-size: 1.1em; font-weight: bold;
                border: none; border-radius: 6px; cursor: pointer;
                background: #27ae60; color: #fff; }
  .btn-discard:hover { background: #e74c3c; }
  .btn-zero:hover    { background: #2ecc71; }
  #keys { font-size: .75em; color: #666; margin-top: 12px; }
  #done { font-size: 1.4em; margin-top: 60px; color: #4CAF50; display: none; }
</style>
</head>
<body>
<div id="info">Loading…</div>
<div id="fname"></div>
<img id="photo" src="" alt="">
<div id="btns">
  <button class="btn-discard" onclick="grade('discard')">Discard (d)</button>
  <button class="btn-zero"    onclick="grade('zero')">Zero Follicles (k)</button>
</div>
<div id="keys">d: Discard &nbsp;|&nbsp; k: Zero Follicles</div>
<div id="done"></div>

<script>
async function load() {
  const d = await fetch('/current').then(r => r.json());
  if (d.done) {
    document.getElementById('photo').style.display = 'none';
    document.getElementById('btns').style.display  = 'none';
    document.getElementById('done').style.display  = 'block';
    document.getElementById('done').textContent    = d.message;
    document.getElementById('info').textContent    = '';
    document.getElementById('fname').textContent   = '';
    return;
  }
  document.getElementById('info').textContent  =
    (d.idx + 1) + ' / ' + d.total + '  (' + d.seen + ' already reviewed)';
  document.getElementById('fname').textContent = d.filename;
  document.getElementById('photo').src =
    '/file?p=' + encodeURIComponent(d.image_path) + '&t=' + Date.now();
}

async function grade(label) {
  await fetch('/grade', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ label }),
  });
  load();
}

document.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
  if (e.key === 'd') grade('discard');
  if (e.key === 'k') grade('zero');
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
    idx   = STATE["idx"]
    items = STATE["items"]
    if idx >= len(items):
        return jsonify({
            "done":    True,
            "message": f"All {len(items)} images reviewed! Results in zero_follicles/",
        })
    img_path = items[idx]
    return jsonify({
        "done":       False,
        "idx":        idx,
        "total":      len(items),
        "seen":       STATE["seen_count"],
        "filename":   os.path.basename(img_path),
        "image_path": img_path,
    })


@app.route("/file")
def serve_file():
    path = request.args.get("p", "")
    if not os.path.isfile(path):
        return "not found", 404
    return send_file(path)


@app.route("/grade", methods=["POST"])
def grade():
    label = request.get_json().get("label")
    idx   = STATE["idx"]
    if idx < len(STATE["items"]):
        img_path = STATE["items"][idx]
        if label == "zero":
            append_row(KEEP_CSV, img_path)
        else:
            append_row(DISCARD_CSV, img_path)
        STATE["idx"] += 1
    return jsonify({"status": "ok"})


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    shuffled = get_or_create_shuffled_list()
    seen     = load_seen_set()
    items    = [p for p in shuffled if p not in seen]

    print(f"Total TF-negative : {len(shuffled)}")
    print(f"Already reviewed  : {len(seen)}")
    print(f"Remaining         : {len(items)}")
    print("Open http://localhost:5000")

    STATE.update({"items": items, "idx": 0, "seen_count": len(seen)})
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
