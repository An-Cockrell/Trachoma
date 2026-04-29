import numpy as np
from PIL import Image, ImageDraw
import os
import json
import cv2 as cv
import math
import sys
import torch
from torchvision import transforms
sys.path.insert(0, "/home/Trachoma/annotated_data/GradableAreaData")
from Training_lightning_GradableArea_UNet import GradableAreaUNet


include_pos = True

GA_PATH = "/home/Trachoma/annotated_data/GradableAreaData/Checkpoints/GradableArea_MobileNetV2_UNet/last.ckpt"
GA_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

manny_image_dir = "/home/Trachoma/annotated_data/vgg_new_data/manny_images/"
chris_image_dir = "/home/Trachoma/annotated_data/vgg_new_data/chris_images/"
clara_image_dir = "/home/Trachoma/annotated_data/vgg_new_data/clara_images/"

manny_ann_json = "/home/Trachoma/annotated_data/vgg_new_data/manny_images_annotations.json"
chris_ann_json = "/home/Trachoma/annotated_data/vgg_new_data/chris_images_annotations.json"
clara_ann_json = "/home/Trachoma/annotated_data/vgg_new_data/clara_images_annotations.json"

foll_mask_save_dir = "/home/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/RawMasks/"
foll_image_save_dir = "/home/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/RawImages/"
ga_mask_save_dir   = "/home/Trachoma/annotated_data/GradableAreaData/FollicleDetectionTest/Masks/"
ga_image_save_dir  = "/home/Trachoma/annotated_data/GradableAreaData/FollicleDetectionTest/Images/"

os.makedirs(foll_mask_save_dir, exist_ok=True)
os.makedirs(foll_image_save_dir, exist_ok=True)
os.makedirs(ga_mask_save_dir, exist_ok=True)
os.makedirs(ga_image_save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ga_model = GradableAreaUNet.load_from_checkpoint(GA_PATH, encoder="mobilenet_v2", lr=1e-3)
ga_model.eval()
ga_model = ga_model.to(device)

_ga_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    GA_NORM,
])

image_dirs = [manny_image_dir, chris_image_dir, clara_image_dir]
annotation_files = [manny_ann_json, chris_ann_json, clara_ann_json]

num_mask = 0


def get_ga_mask_and_width(img_pil, ann_ga_width):
    """Return (mask_orig, ga_width). mask_orig is uint8 {0,1} at original resolution.
    ga_width falls back to ann_ga_width if the model finds no gradable area."""
    img_tensor = _ga_preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = ga_model(img_tensor)
    mask_512 = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    W, H = img_pil.size
    mask_orig = cv.resize(mask_512, (W, H), interpolation=cv.INTER_NEAREST)
    x_cols = np.where(mask_orig.sum(axis=0) > 0)[0]
    if len(x_cols) == 0:
        print(f"  GA model found no gradable area — using annotation ga_width={ann_ga_width}")
        return mask_orig, ann_ga_width
    return mask_orig, int(x_cols.max() - x_cols.min())


def save_image_and_masks(img, im_filename, def_foll, pos_foll, ann_ga_width):
    """Run GA model, draw follicle mask, and save all outputs."""
    ga_mask, ga_width = get_ga_mask_and_width(img, ann_ga_width)
    print(f"  [{im_filename}] ga_width={ga_width}")

    radius = math.ceil(ga_width / 80) if ga_width > 0 else 5

    foll_mask = Image.new("1", img.size, 0)
    foll_draw = ImageDraw.Draw(foll_mask)
    follicles = def_foll + pos_foll if include_pos else def_foll
    for coord in follicles:
        foll_draw.ellipse(
            (coord[0] - radius, coord[1] - radius, coord[0] + radius, coord[1] + radius),
            fill=255,
        )

    mask_filename = os.path.splitext(im_filename)[0] + ".png"
    img.save(f"{foll_image_save_dir}{im_filename}")
    foll_mask.save(f"{foll_mask_save_dir}{mask_filename}")
    Image.fromarray(ga_mask * 255).save(f"{ga_mask_save_dir}{mask_filename}")
    img.save(f"{ga_image_save_dir}{im_filename}")


# ── VGG new data (manny / chris / clara) ──────────────────────────────────────

for image_dir, annotation_file in zip(image_dirs, annotation_files):
    with open(annotation_file, "r") as f:
        ann_dict = json.load(f)

    for name, annotation in ann_dict.items():
        skip_image = False
        im_filename = annotation["filename"]
        pos_foll = []
        def_foll = []
        ann_ga_width = 0

        img_file = image_dir + im_filename
        img = Image.open(img_file)

        for region in annotation["regions"]:
            shape_attributes = region["shape_attributes"]
            region_attributes = region["region_attributes"]
            shape_type = shape_attributes["name"]
            class_type = region_attributes["type"]
            if (class_type == "GrArea") or (shape_type == "polygon"):
                try:
                    x_coords = shape_attributes["all_points_x"]
                    y_coords = shape_attributes["all_points_y"]
                    ann_ga_width = abs(np.max(x_coords) - np.min(x_coords))
                except Exception as e:
                    print(im_filename, "Gradable Area Error", e)
                    skip_image = True
                    break
            else:
                try:
                    coords = (shape_attributes["cx"], shape_attributes["cy"])
                    if class_type == "DefFol":
                        def_foll.append(coords)
                    elif class_type == "PosFol":
                        pos_foll.append(coords)
                    else:
                        print(class_type)
                except Exception as e:
                    print(im_filename, "Follicle Error", e)
                    skip_image = True
                    break

        if skip_image:
            continue

        num_mask += len(def_foll) + len(pos_foll)
        save_image_and_masks(img, im_filename, def_foll, pos_foll, ann_ga_width)


# ── Supervisely MoreImages (Chris + Lindsay, 73 shared images each) ───────────
# Annotation format: per-image JSON in ann/ with objects having classTitle:
#   "Grading Area"       → polygon gradable area (points.exterior = [[x,y], ...])
#   "Definite Follicle"  → point (DefFol)
#   "Ambiguous Follicle" → point (PosFol)
# Source images live in clean_images/ named imageXXX.jpg.
# Both annotators annotated the same 73 images; saved with _chris/_lindsay suffix
# to keep both perspectives as separate training samples.

supervisely_sources = [
    (
        "/home/Trachoma/annotated_data/data from Supervisely task/MoreImages/UCSF TF positive Chris/ann/",
        "/home/Trachoma/annotated_data/FollicleDetection/clean_images/",
        "chris",
    ),
    (
        "/home/Trachoma/annotated_data/data from Supervisely task/MoreImages/UCSF TF positive Lindsay/ann/",
        "/home/Trachoma/annotated_data/FollicleDetection/clean_images/",
        "lindsay",
    ),
]

import glob as _glob

for ann_dir, img_dir, annotator in supervisely_sources:
    for ann_file in sorted(_glob.glob(ann_dir + "*.json")):
        base_img = os.path.basename(ann_file).replace(".json", "")  # e.g. image101.jpg
        img_stem = os.path.splitext(base_img)[0]                    # e.g. image101
        img_path = os.path.join(img_dir, base_img)

        if not os.path.exists(img_path):
            print(f"  Missing source image: {img_path} — skipping")
            continue

        with open(ann_file, "r") as f:
            ann = json.load(f)

        skip_image = False
        def_foll, pos_foll = [], []
        ann_ga_width = 0

        for obj in ann.get("objects", []):
            title = obj["classTitle"]
            pts = obj["points"]["exterior"]
            if title == "Grading Area":
                try:
                    xs = [p[0] for p in pts]
                    ann_ga_width = abs(max(xs) - min(xs))
                except Exception as e:
                    print(base_img, "Grading Area Error", e)
                    skip_image = True
                    break
            elif title == "Definite Follicle":
                def_foll.append((pts[0][0], pts[0][1]))
            elif title == "Ambiguous Follicle":
                pos_foll.append((pts[0][0], pts[0][1]))

        if skip_image:
            continue

        img = Image.open(img_path)
        # Save with annotator suffix to keep both annotations as separate training samples
        save_filename = f"0_{img_stem}_{annotator}.jpg"
        num_mask += len(def_foll) + len(pos_foll)
        save_image_and_masks(img, save_filename, def_foll, pos_foll, ann_ga_width)
