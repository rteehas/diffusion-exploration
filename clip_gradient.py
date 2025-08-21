#!/usr/bin/env python3
"""
Compute the gradient of the CLIP similarity score with respect to the input image pixels.
Requires:
    pip install git+https://github.com/openai/CLIP.git torch pillow
"""

import torch
import clip
from PIL import Image
from pathlib import Path


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and preprocessing pipeline
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
MODEL.eval()  # turn off dropout


def grad_clip_score(img, prompt):
    """Return (score, d(score)/d(image)) for given image file and text prompt.

    img_path:
        Path to an RGB image file.
    prompt:
        Natural‑language text prompt.
    """
    # 1. Load & preprocess image; keep gradient.
    # img = Image.open(image_path).convert("RGB")
    img_tensor = PREPROCESS(img).unsqueeze(0).to(DEVICE).requires_grad_(True)  # (1,3,H,W)

    # 2. Encode text (no gradient needed).
    text_tokens = clip.tokenize([prompt]).to(DEVICE)

    # 3. Forward pass through CLIP encoders.
    img_feat = MODEL.encode_image(img_tensor)
    txt_feat = MODEL.encode_text(text_tokens)

    # 4. Normalize embeddings and scale.
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    logit_scale = MODEL.logit_scale.exp()

    # 5. Similarity score (scalar).
    score = (logit_scale * img_feat @ txt_feat.T).squeeze()

    # 6. Back‑propagate to image.
    (-score).backward()  # negative for gradient ascent direction
    grad = img_tensor.grad.detach().cpu()  # (1,3,H,W)

    return score.item(), grad


if __name__ == "__main__":
    import argparse, numpy as np

    parser = argparse.ArgumentParser(description="Gradient of CLIP score")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("prompt", help="Text prompt")
    args = parser.parse_args()

    s, g = grad_clip_score(args.image, args.prompt)
    print(f"Score: {s:.4f}")
    print(f"Gradient magnitude: {float(g.norm()):.4f}")

    # Optional: save gradient as numpy for external use.
    np.save(Path(args.image).with_suffix(".clip_grad.npy"), g.numpy())
