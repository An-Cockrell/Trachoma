"""
visualize_enhancement.py

Standalone script to visually compare original vs. follicle-enhanced images.
Follicles are whitish (low saturation, low a*) against pinkish background
(high saturation, high a*). The enhancement pipeline exploits this contrast.

Usage:
    # Single image — default 2-panel (original + enhanced)
    python visualize_enhancement.py --image path/to/image.jpg

    # Show all color/contrast scheme comparisons in a grid
    python visualize_enhancement.py --image path/to/image.jpg --schemes

    # Directory of images (picks up to --n random ones)
    python visualize_enhancement.py --dir path/to/img_dir --n 8

    # Save outputs instead of displaying
    python visualize_enhancement.py --dir path/to/img_dir --save_dir ./enhanced_previews

    # Save scheme grids
    python visualize_enhancement.py --dir path/to/img_dir --schemes --save_dir ./scheme_previews
"""

import argparse
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Glare removal
# ---------------------------------------------------------------------------


GLARE_SIGMA = 2.5  # how many robust-std above the median counts as glare
GLARE_FLOOR = 210  # never flag pixels below this brightness regardless of statistics


def remove_glare(img_rgb: np.ndarray, inpaint_radius: int = 5) -> np.ndarray:
    """
    Detects specular glare using robust statistics and inpaints with TELEA.

    Threshold = median(V) + GLARE_SIGMA * robust_std(V), where robust_std is
    derived from the MAD (median absolute deviation). Unlike mean/std, median
    and MAD are unaffected by the glare pixels themselves, so the glare cannot
    inflate its own threshold and escape detection.

    A hard floor (GLARE_FLOOR) prevents over-flagging on uniformly bright images
    where even normal tissue might otherwise exceed the sigma threshold.

    Args:
        img_rgb:        uint8 H×W×3 RGB image
        inpaint_radius: neighborhood radius for inpainting (default 5px)
    Returns:
        uint8 H×W×3 RGB image with glare regions filled in
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    V = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 2].astype(np.float32)

    median_v = np.median(V)
    mad_v = np.median(np.abs(V - median_v))
    robust_std = 1.4826 * mad_v  # equivalent to std for a normal distribution

    threshold = median_v + GLARE_SIGMA * robust_std
    threshold = max(threshold, GLARE_FLOOR)

    glare_mask = (V >= threshold).astype(np.uint8) * 255

    # Dilate to catch the blown-out halo around each glare spot
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    glare_mask = cv2.dilate(glare_mask, kernel, iterations=2)

    inpainted_bgr = cv2.inpaint(img_bgr, glare_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Core enhancement functions
# ---------------------------------------------------------------------------


def build_follicle_channel(img_rgb: np.ndarray) -> np.ndarray:
    """
    Builds a single-channel follicle probability map (uint8, 0–255).
    Higher values = more follicle-like (whitish, low saturation, low redness).

    The map is dynamic: it uses the image's own color statistics to decide
    how much weight to give each cue, so it adapts to different cameras and
    lighting conditions automatically.
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    S = img_hsv[:, :, 1].astype(np.float32)  # saturation  0–255
    V = img_hsv[:, :, 2].astype(np.float32)  # brightness  0–255
    a_star = img_lab[:, :, 1].astype(np.float32)  # red-green   0–255 (128 = neutral)

    # --- Cue 1: Whiteness  (low saturation × high brightness) ---
    whiteness = ((255.0 - S) / 255.0) * (V / 255.0)
    whiteness = (whiteness * 255).astype(np.uint8)

    # --- Cue 2: Non-redness  (invert & amplify the a* axis) ---
    # Pink tissue sits well above 128; follicles sit near or below 128.
    # Amplify by 2× before inverting so small differences become large ones.
    inv_a = np.clip(255.0 - (a_star - 128.0) * 2.0, 0, 255).astype(np.uint8)

    # --- Dynamic per-image weighting ---
    # If the image is very pink (mean a* >> 128), trust whiteness more.
    mean_a = float(np.mean(a_star))
    pink_weight = np.clip(
        (mean_a - 128.0) / 40.0, 0.0, 1.0
    )  # 0 = neutral, 1 = very pink
    w_white = 0.5 + 0.3 * pink_weight  # 0.50 → 0.80
    w_red = 1.0 - w_white  # 0.50 → 0.20

    combined = cv2.addWeighted(whiteness, w_white, inv_a, w_red, 0)

    # --- CLAHE on combined map to enhance local blob contrast ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(combined)

    # --- Difference-of-Gaussians bandpass at follicle-typical scales ---
    # Suppresses uniform background and noise; keeps blobs ~5–20 px radius.
    fine = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    coarse = cv2.GaussianBlur(enhanced, (21, 21), 6.0)
    dog = cv2.subtract(fine, coarse)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)

    # --- Final: blend CLAHE map + DoG ---
    follicle_ch = cv2.addWeighted(enhanced, 0.6, dog, 0.4, 0)
    return follicle_ch


def enhance_image(img_rgb: np.ndarray) -> np.ndarray:
    """
    Follicle contrast enhancement (expects glare already removed).
    Returns a natural-looking RGB image with follicle regions made brighter
    and more white.
    """
    follicle_ch = build_follicle_channel(img_rgb)
    follicle_norm = follicle_ch.astype(np.float32) / 255.0

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Push brightness up in follicle regions
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] + follicle_norm * 60, 0, 255)
    # Reduce saturation in follicle regions (make them look whiter)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * (1 - follicle_norm * 0.6), 0, 255)

    result = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return result


# ---------------------------------------------------------------------------
# Color/contrast scheme functions
# Each returns an RGB uint8 image (or grayscale displayed via imshow).
# ---------------------------------------------------------------------------


def scheme_original(img_rgb: np.ndarray) -> np.ndarray:
    return img_rgb


def scheme_enhanced(img_rgb: np.ndarray) -> np.ndarray:
    """Current full pipeline."""
    return enhance_image(img_rgb)


def scheme_red_suppressed(img_rgb: np.ndarray) -> np.ndarray:
    """
    False color: halve the red channel, boost green and blue.
    Follicles (whitish, balanced RGB) stay roughly white/cyan.
    Background (reddish) turns much dimmer, pushing follicles forward.
    """
    img = img_rgb.astype(np.float32)
    out = np.zeros_like(img)
    out[:, :, 0] = np.clip(img[:, :, 0] * 0.3, 0, 255)   # R: suppress
    out[:, :, 1] = np.clip(img[:, :, 1] * 1.4, 0, 255)   # G: boost
    out[:, :, 2] = np.clip(img[:, :, 2] * 1.6, 0, 255)   # B: boost more
    return out.astype(np.uint8)


def scheme_gr_ratio(img_rgb: np.ndarray) -> np.ndarray:
    """
    Green-to-Red ratio map (grayscale).
    Follicles are whitish → balanced channels → G/R ≈ 1 → bright.
    Reddish tissue → G/R < 1 → dark.
    Result normalized to 0–255 and CLAHE-enhanced.
    """
    R = img_rgb[:, :, 0].astype(np.float32) + 1.0  # +1 avoids div-by-zero
    G = img_rgb[:, :, 1].astype(np.float32)
    ratio = np.clip(G / R * 128.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    ratio = clahe.apply(ratio)
    return cv2.cvtColor(ratio, cv2.COLOR_GRAY2RGB)


def scheme_br_diff(img_rgb: np.ndarray) -> np.ndarray:
    """
    Blue minus Red difference map.
    Follicles (balanced/bluish white) → B-R ≈ 0 or positive → bright.
    Reddish tissue → B-R negative → dark.
    Shifted to 0–255 range and CLAHE-enhanced.
    """
    R = img_rgb[:, :, 0].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)
    diff = np.clip((B - R) / 2.0 + 128.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    diff = clahe.apply(diff)
    return cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)


def scheme_lab_a_inverted(img_rgb: np.ndarray) -> np.ndarray:
    """
    Pure LAB a* channel, inverted.
    a* encodes red-green axis. Reddish tissue = high a*; follicles = low a*.
    Inverting makes follicles bright white against a dark background.
    CLAHE applied for local contrast boost.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a = lab[:, :, 1].astype(np.float32)  # 0–255, 128 = neutral
    inv_a = np.clip(255.0 - a, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    inv_a = clahe.apply(inv_a)
    return cv2.cvtColor(inv_a, cv2.COLOR_GRAY2RGB)


def scheme_ycbcr_cb(img_rgb: np.ndarray) -> np.ndarray:
    """
    YCbCr Cb (blue-difference) channel.
    White/bluish follicles → high Cb. Red tissue → low Cb.
    CLAHE applied.
    """
    ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    cb = ycbcr[:, :, 2]  # Cb channel
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cb = clahe.apply(cb)
    return cv2.cvtColor(cb, cv2.COLOR_GRAY2RGB)


def scheme_saturation_inv(img_rgb: np.ndarray) -> np.ndarray:
    """
    Inverted HSV saturation channel.
    Follicles (low saturation/whitish) → high value here.
    Background (vivid pink/red) → low value.
    Brightness-weighted to further separate follicles from dark low-sat areas.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    # Weight inverted saturation by brightness so dark low-sat pixels don't score high
    whiteness = ((255.0 - S) / 255.0) * (V / 255.0)
    whiteness = cv2.normalize(whiteness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    whiteness = clahe.apply(whiteness)
    return cv2.cvtColor(whiteness, cv2.COLOR_GRAY2RGB)


def scheme_heatmap_overlay(img_rgb: np.ndarray) -> np.ndarray:
    """
    Follicle probability heatmap (JET colormap) blended over the original.
    Hot = high follicle probability. Useful for seeing where the detector fires.
    """
    img_no_glare = remove_glare(img_rgb)
    fmap = build_follicle_channel(img_no_glare)
    jet = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_no_glare, 0.55, jet_rgb, 0.45, 0)
    return overlay


def scheme_desat_false_color(img_rgb: np.ndarray) -> np.ndarray:
    """
    Desaturate the image heavily then colorize the red channel as blue.
    Pink tissue loses its color advantage; follicles appear as distinct
    cool (blue-grey) blobs against a neutral background.
    R→B, G→G, B→R swap + aggressive desaturation of the original.
    """
    # Convert to HSV, reduce saturation dramatically
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.25  # 75% desaturation
    desat = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    # Swap R and B to shift remaining redness to cool tones
    out = desat.copy()
    out[:, :, 0] = desat[:, :, 2]  # R ← B
    out[:, :, 2] = desat[:, :, 0]  # B ← R
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# All schemes registry
# ---------------------------------------------------------------------------

SCHEMES = [
    ("Original",            scheme_original),
    ("Current Enhanced",    scheme_enhanced),
    ("Red Suppressed",      scheme_red_suppressed),
    ("G/R Ratio",           scheme_gr_ratio),
    ("B−R Difference",      scheme_br_diff),
    ("LAB a* Inverted",     scheme_lab_a_inverted),
    ("YCbCr Cb",            scheme_ycbcr_cb),
    ("Saturation Inv",      scheme_saturation_inv),
    ("Heatmap Overlay",     scheme_heatmap_overlay),
    ("Desat + BGR Swap",    scheme_desat_false_color),
]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_comparison(img_rgb: np.ndarray, title: str = "", save_path: str = None):
    """Side-by-side: original | enhanced."""
    img_no_glare = remove_glare(img_rgb)
    enhanced = enhance_image(img_no_glare)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title or "Follicle Enhancement", fontsize=13)

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(enhanced)
    axes[1].set_title("Enhanced")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {save_path}")
    else:
        plt.show()


def plot_schemes(img_rgb: np.ndarray, title: str = "", save_path: str = None):
    """
    Grid showing all color/contrast schemes for a single image.
    Layout: 2 rows × 5 cols (10 schemes).
    """
    n = len(SCHEMES)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    fig.suptitle(title or "Color/Contrast Scheme Comparison", fontsize=14)
    axes = axes.flatten()

    img_rgb = remove_glare(img_rgb)

    for i, (name, fn) in enumerate(SCHEMES):
        try:
            result = fn(img_rgb)
        except Exception as e:
            result = np.zeros_like(img_rgb)
            print(f"  [WARN] {name} failed: {e}")
        axes[i].imshow(result)
        axes[i].set_title(name, fontsize=9)
        axes[i].axis("off")

    # Hide any unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def collect_images(directory: str, extensions=(".jpg", ".jpeg", ".png")) -> list:
    paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(extensions) and "mask" not in root.lower():
                paths.append(os.path.join(root, f))
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Visualize follicle enhancement pipeline"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to a single image file")
    src.add_argument("--dir", help="Directory to search for images")

    parser.add_argument(
        "--n",
        type=int,
        default=6,
        help="Number of random images to sample from --dir (default: 6)",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help="If set, save comparison PNGs here instead of displaying",
    )
    parser.add_argument(
        "--schemes",
        action="store_true",
        help="Show all color/contrast schemes in a grid instead of just original vs enhanced",
    )
    args = parser.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    plot_fn = plot_schemes if args.schemes else plot_comparison

    if args.image:
        img = load_rgb(args.image)
        name = os.path.basename(args.image)
        print(f"Processing: {name}")
        suffix = "_schemes.png" if args.schemes else f"_enhanced{os.path.splitext(name)[1]}"
        save_path = (
            os.path.join(args.save_dir, os.path.splitext(name)[0] + suffix)
            if args.save_dir
            else None
        )
        plot_fn(img, title=name, save_path=save_path)

    else:  # --dir
        all_images = collect_images(args.dir)
        if not all_images:
            print(f"No images found in: {args.dir}")
            sys.exit(1)

        sample = random.sample(all_images, min(args.n, len(all_images)))
        print(f"Found {len(all_images)} images, showing {len(sample)}")

        for path in sample:
            name = os.path.basename(path)
            print(f"  Processing: {name}")
            img = load_rgb(path)
            suffix = "_schemes.png" if args.schemes else f"_enhanced{os.path.splitext(name)[1]}"
            save_path = (
                os.path.join(args.save_dir, os.path.splitext(name)[0] + suffix)
                if args.save_dir
                else None
            )
            plot_fn(img, title=name, save_path=save_path)


if __name__ == "__main__":
    main()
