"""
Export GA and Follicle UNet models to ONNX format.
Run from the same directory as MultipleFollicleTest_TF_prediction.py.
Output: ga_model.onnx and follicle_model.onnx in the same directory.
"""

import os
import sys
import torch

# ── Same path setup as MultipleFollicleTest_TF_prediction.py ──────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../GradableAreaData"))
from Training_lightning_GradableArea_UNet import GradableAreaUNet
from Training_lightning_MultipleFollicle_UNet_Pretrained import FollicleUNet

GA_PATH       = "../GradableAreaData/Checkpoints/GradableArea_MobileNetV2_UNet/last.ckpt"
FOLLICLE_PATH = "Checkpoints/MultipleFollicle_EfficientNetB4_UNet_Pretrained/last.ckpt"

OUT_GA       = "ga_model.onnx"
OUT_FOLLICLE = "follicle_model.onnx"

# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading GA model...")
ga_model = GradableAreaUNet.load_from_checkpoint(GA_PATH, encoder="mobilenet_v2", lr=1e-3)
ga_model.eval().cpu()

print("Loading Follicle model...")
follicle_model = FollicleUNet.load_from_checkpoint(FOLLICLE_PATH)
follicle_model.eval().cpu()

# ── Export to ONNX ────────────────────────────────────────────────────────────

example_input = torch.rand(1, 3, 512, 512)  # CPU, matches models

print("Exporting GA model to ONNX...")
with torch.no_grad():
    torch.onnx.export(
        ga_model,
        example_input,
        OUT_GA,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
print(f"  Saved: {OUT_GA}")

print("Exporting Follicle model to ONNX...")
with torch.no_grad():
    torch.onnx.export(
        follicle_model,
        example_input,
        OUT_FOLLICLE,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
print(f"  Saved: {OUT_FOLLICLE}")

# ── Verify outputs ────────────────────────────────────────────────────────────

import onnxruntime as ort
import numpy as np

print("\nVerifying GA model...")
sess = ort.InferenceSession(OUT_GA)
out = sess.run(None, {"input": example_input.numpy()})
print(f"  GA output shape: {out[0].shape}")  # expect (1, 1, 512, 512)

print("Verifying Follicle model...")
sess = ort.InferenceSession(OUT_FOLLICLE)
out = sess.run(None, {"input": example_input.numpy()})
print(f"  Follicle output shape: {out[0].shape}")  # expect (1, 1, 512, 512)

print("\nDone. Copy ga_model.onnx and follicle_model.onnx to your Android project's assets/ folder.")
