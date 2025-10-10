import torch
import numpy as np
from PIL import Image
import clip
from .utils import get_device


class ImageEmbedder:
    def __init__(self, clip_model="ViT-B/32", device=None):
        if device is None:
            device = get_device()
        self.model, self.preprocess = clip.load(clip_model, device=device)
        self.device = device

    def embedImage(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image_input)[0]
        return embedding.cpu()
