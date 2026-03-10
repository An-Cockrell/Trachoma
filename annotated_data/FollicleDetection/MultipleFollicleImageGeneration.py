import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import glob
import os
import json
import cv2 as cv
from scipy import spatial
import math


def estimate_follicle_radius_hough(img_gray, cx, cy, ga_radius, patch_factor=3):
    """
    Estimate the radius of a single follicle using Hough Circle Transform on a
    local patch around the annotated center point (cx, cy).

    Since annotations are center points only (no boundary), this extracts the
    actual circular structure from the image. Falls back to ga_radius if no
    circle is confidently detected.

    Args:
        img_gray: Grayscale numpy array (H, W) uint8 of the full image.
        cx, cy:   Annotated follicle center in image coordinates.
        ga_radius: GA-width-based fallback radius estimate.
        patch_factor: Half-patch size = patch_factor * ga_radius.

    Returns:
        Estimated radius (int).
    """
    half = int(patch_factor * ga_radius)
    h, w = img_gray.shape

    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half), min(h, cy + half)
    patch = img_gray[y1:y2, x1:x2]

    if patch.size == 0 or patch.shape[0] < 6 or patch.shape[1] < 6:
        return ga_radius

    # Enhance local contrast then smooth before Hough
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    patch = clahe.apply(patch)
    patch = cv.GaussianBlur(patch, (3, 3), 0)

    min_r = max(3, ga_radius // 2)
    max_r = min(ga_radius * 2, half - 2)

    if min_r >= max_r:
        return ga_radius

    circles = cv.HoughCircles(
        patch,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=patch.shape[0],  # expect at most one follicle per patch
        param1=50,
        param2=12,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        # Pick the circle closest to the patch center
        pcx, pcy = (x2 - x1) // 2, (y2 - y1) // 2
        best = min(circles, key=lambda c: (c[0] - pcx) ** 2 + (c[1] - pcy) ** 2)
        return int(best[2])

    return ga_radius



# dir1 = '/media/dsocia22/T7/Trachoma/annotated_data/data from Supervisely task/MoreImages/UCSF TF positive Chris'
# dir2 = '/media/dsocia22/T7/Trachoma/annotated_data/data from Supervisely task/MoreImages/UCSF TF positive Lindsay'

# PosFol policy: only definite follicles (DefFol) are used for training masks.
# Possible follicles (PosFol) are excluded because their ambiguity would introduce
# noisy labels. If this changes, regenerate all masks before retraining.
include_pos = False

crop_image = True

manny_image_dir = "/home/Trachoma/annotated_data/vgg_new_data/manny_images/"

chris_image_dir = "/home/Trachoma/annotated_data/vgg_new_data/chris_images/"

clara_image_dir = "/home/Trachoma/annotated_data/vgg_new_data/clara_images/"


manny_ann_json = (
    "/home/Trachoma/annotated_data/vgg_new_data/manny_images_annotations.json"
)

chris_ann_json = (
    "/home/Trachoma/annotated_data/vgg_new_data/chris_images_annotations.json"
)

clara_ann_json = (
    "/home/Trachoma/annotated_data/vgg_new_data/clara_images_annotations.json"
)


grad_area_save_dir = "/home/Trachoma/annotated_data/GradableAreaData/new_data/Mask/"

grad_area_image_save_dir = (
    "/home/Trachoma/annotated_data/GradableAreaData/new_data/Images/"
)


if include_pos:
    foll_mask_save_dir = "/home/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/new_data/Masks/all_follicles/"
else:
    foll_mask_save_dir = "/home/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/new_data/Masks/def_follicles/"


foll_image_save_dir = "/home/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/new_data/Images/"

# foll_mask_save_dir = '/home/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/image_gen_examples/Masks/'

# foll_image_save_dir = '/home/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/image_gen_examples/Images/'

# foll_annotated_image_save_dir = '/home/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/image_gen_examples/annotated/'

os.makedirs(grad_area_save_dir, exist_ok=True)

os.makedirs(grad_area_image_save_dir, exist_ok=True)

os.makedirs(foll_mask_save_dir, exist_ok=True)

os.makedirs(foll_image_save_dir, exist_ok=True)

# os.makedirs(foll_annotated_image_save_dir, exist_ok=True)

image_dirs = [manny_image_dir, chris_image_dir, clara_image_dir]
annotation_files = [manny_ann_json, chris_ann_json, clara_ann_json]

num_mask = 0

failed_image_count = 0


for image_dir, annotation_file in zip(image_dirs, annotation_files):
    # if p == 0:
    #     name = 'chris'
    # else:
    #     name = 'lindsay'
    # img_num_save = 0
    # data_dir = df + '/ann/'
    # image_dir = d + '/img/'
    with open(annotation_file, "r") as f:
        ann_dict = json.load(f)

    # num_images = 0
    for name, annotation in ann_dict.items():

        # if (num_images >= 5):
        #     break
        skip_image = False  # flag for skipping images that raise exceptions
        im_filename = annotation["filename"]
        size = annotation["size"]
        # size = os.path.getsize(data_file)
        pos_foll = []
        def_foll = []

        img_file = image_dir + im_filename

        img = Image.open(img_file)
        # img2 = Image.open(img_file)
        # img3 = Image.open(img_file)
        # # print(img.size)
        # sizes.append(img.size)

        # if img.size == (1024, 680):
        #     radius = 10
        # elif img.size == (2048, 1536):
        #     radius == 15
        # elif img.size == (2240, 1488):
        #     radius = 15
        # elif img.size == (3008, 2000):
        #     radius = 20
        # elif img.size == (3872, 2592):
        #     radius = 25
        # elif img.size == (4288, 2848):
        #     radius = 30
        # else:
        #     print(f"{im_filename} has unknown image size: {img.size}")

        poly = Image.new("1", img.size, 0)
        pdraw = ImageDraw.Draw(poly)
        radius = 0

        for region in annotation["regions"]:
            shape_attributes = region["shape_attributes"]
            region_attributes = region["region_attributes"]
            shape_type = shape_attributes["name"]
            class_type = region_attributes["type"]
            if (class_type == "GrArea") or (shape_type == "polygon"):
                try:
                    x_coords = shape_attributes["all_points_x"]
                    y_coords = shape_attributes["all_points_y"]
                    min_x = np.min(x_coords)
                    max_x = np.max(x_coords)
                    ga_width = abs(max_x - min_x)
                    # assuming a standard gradable area width of 20mm: 0.25 mm radius = gradable area width / 80
                    radius = math.ceil(ga_width / 80)
                    coords = list(zip(x_coords, y_coords))
                    pdraw.polygon(coords, fill=255)
                    poly.save(grad_area_save_dir + im_filename)
                    img.save(grad_area_image_save_dir + im_filename)
                except Exception as e:
                    print(im_filename)
                    print("Gradable Area Error")
                    print(e)
                    skip_image = True
                    break
            else:
                try:
                    coords = (shape_attributes["cx"], shape_attributes["cy"])
                    if class_type == "DefFol":
                        color = "green"
                        def_foll.append(coords)
                    elif class_type == "PosFol":
                        color = "blue"
                        pos_foll.append(coords)
                    else:
                        print(class_type)
                except Exception as e:
                    print(im_filename)
                    print("Follicle Error")
                    print(e)
                    skip_image = True
                    break

        if skip_image:
            continue

        num_mask = num_mask + len(def_foll) + len(pos_foll)

        # Convert to grayscale once for Hough radius estimation
        img_gray = cv.cvtColor(np.asarray(img, dtype=np.uint8), cv.COLOR_RGB2GRAY)

        foll_mask = Image.new("1", img.size, 0)
        foll_draw = ImageDraw.Draw(foll_mask)

        if include_pos:
            follicles = def_foll + pos_foll
        else:
            follicles = def_foll

        for coord in follicles:
            # Estimate per-follicle radius from image content; fall back to GA-based radius
            foll_radius = estimate_follicle_radius_hough(img_gray, coord[0], coord[1], radius)
            foll_draw.ellipse(
                (
                    coord[0] - foll_radius,
                    coord[1] - foll_radius,
                    coord[0] + foll_radius,
                    coord[1] + foll_radius,
                ),
                fill=255,
            )
        if crop_image:
            #CROP IMAGE TO GRADABLE AREA!
            ga_mask = poly.convert("L")
            ga_mask_np = np.asarray(ga_mask, dtype=np.uint8)
            ga_bin_mask = (ga_mask_np > 0).astype(np.uint8)

            # 2) Multiply mask by image (outside becomes black)
            img_np = np.asarray(img, dtype=np.uint8)     # shape (H, W, 3)
            masked_np = img_np * ga_bin_mask[:, :, None]        # broadcast to 3 channels

            # Back to PIL
            masked_img = Image.fromarray(masked_np, mode="RGB")
            masked_img.save(f"{foll_image_save_dir}{im_filename}")
        else:
            img.save(f"{foll_image_save_dir}{im_filename}")
        
        foll_mask.save(f"{foll_mask_save_dir}{im_filename}")



        # img = img.convert("RGB")
        # draw = ImageDraw.Draw(img)

        # # Pick which follicles to draw
        # follicles = (def_foll + pos_foll) if include_pos else def_foll

        # r = int(round(radius))  # make sure radius is an int

        # for (x, y) in follicles:
        #     x = int(round(x))
        #     y = int(round(y))
        #     bbox = (x - r, y - r, x + r, y + r)
        #     # Unfilled ellipse, black outline, 2px border (adjust width as you like)
        #     draw.ellipse(bbox, outline=(0, 255, 0), width=5)

        # # Save the annotated image
        # img.save(f"{foll_annotated_image_save_dir}{im_filename}")
        # num_images+=1
