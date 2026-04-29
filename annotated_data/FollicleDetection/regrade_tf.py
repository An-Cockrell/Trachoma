#!/usr/bin/env python3
"""
TF regrading application for error cases.

Combines false_positives.txt and false_negatives.txt into one randomized list
(generated once, stored in regrading/combined_shuffled.txt) and presents each
image for blind regrading.

Output : regrading/new_tf_grades.csv
Hotkeys: k = TF  |  d = NO TF

Open http://localhost:5000
"""

import csv
import os
import random

from flask import Flask, jsonify, request, send_file

app   = Flask(__name__)
STATE = {}

_HERE       = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR    = os.path.join(
    _HERE,
    "follicle_detection_figs",
    "EfficientNetB4_UNet_Pretrained_TF_eval",
)
REGRADE_DIR   = os.path.join(EVAL_DIR, "regrading")
SHUFFLED_FILE = os.path.join(REGRADE_DIR, "combined_shuffled.txt")
GRADES_CSV    = os.path.join(REGRADE_DIR, "new_tf_grades.csv")


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_or_create_shuffled_list():
    if os.path.exists(SHUFFLED_FILE):
        with open(SHUFFLED_FILE) as f:
            items = [line.strip() for line in f if line.strip()]
        print(f"Loaded existing shuffled list: {len(items)} images")
        return items

    fp_file = os.path.join(EVAL_DIR, "false_positives.txt")
    fn_file = os.path.join(EVAL_DIR, "false_negatives.txt")
    with open(fp_file) as f:
        fp_list = [line.strip() for line in f if line.strip()]
    with open(fn_file) as f:
        fn_list = [line.strip() for line in f if line.strip()]

    combined = fp_list + fn_list
    random.shuffle(combined)

    os.makedirs(REGRADE_DIR, exist_ok=True)
    with open(SHUFFLED_FILE, "w") as f:
        for path in combined:
            f.write(path + "\n")

    print(
        f"Generated shuffled list: {len(combined)} images "
        f"({len(fp_list)} FP + {len(fn_list)} FN)"
    )
    return combined


def load_graded_set():
    if not os.path.exists(GRADES_CSV):
        return set()
    with open(GRADES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        return {row["image_path"] for row in reader}


def save_grade(img_path, label):
    os.makedirs(REGRADE_DIR, exist_ok=True)
    write_header = not os.path.exists(GRADES_CSV)
    with open(GRADES_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["image_path", "TF"])
        w.writerow([img_path, 1 if label == "TF" else 0])


# ── HTML ───────────────────────────────────────────────────────────────────────

HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TF Regrader</title>
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
  .btn-no { padding: 12px 52px; font-size: 1.1em; font-weight: bold;
            border: none; border-radius: 6px; cursor: pointer;
            background: #c0392b; color: #fff; }
  .btn-tf { padding: 12px 52px; font-size: 1.1em; font-weight: bold;
            border: none; border-radius: 6px; cursor: pointer;
            background: #27ae60; color: #fff; }
  .btn-no:hover { background: #e74c3c; }
  .btn-tf:hover { background: #2ecc71; }
  #keys { font-size: .75em; color: #666; margin-top: 12px; }
  #done { font-size: 1.4em; margin-top: 60px; color: #4CAF50; display: none; }
</style>
</head>
<body>
<div id="info">Loading…</div>
<div id="fname"></div>
<img id="photo" src="" alt="">
<div id="btns">
  <button class="btn-no" onclick="grade('NO TF')">NO TF</button>
  <button class="btn-tf" onclick="grade('TF')">TF</button>
</div>
<div id="keys">d: NO TF &nbsp;|&nbsp; k: TF</div>
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
  document.getElementById('info').textContent  = (d.idx + 1) + ' / ' + d.total;
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
  if (e.key === 'd') grade('NO TF');
  if (e.key === 'k') grade('TF');
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
    s, idx, items = STATE, STATE["idx"], STATE["items"]
    if idx >= len(items):
        return jsonify({
            "done":    True,
            "message": f"All {len(items)} images graded! Results in regrading/new_tf_grades.csv",
        })
    img_path = items[idx]
    return jsonify({
        "done":       False,
        "idx":        idx,
        "total":      len(items),
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
    if STATE["idx"] < len(STATE["items"]):
        save_grade(STATE["items"][STATE["idx"]], label)
        STATE["idx"] += 1
    return jsonify({"status": "ok"})


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    shuffled = get_or_create_shuffled_list()
    graded   = load_graded_set()
    items    = [p for p in shuffled if p not in graded]

    print(f"Already graded : {len(graded)}")
    print(f"Remaining      : {len(items)}")
    print("Open http://localhost:5000")

    STATE.update({"items": items, "idx": 0})
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
