# PaddleOCR handles preprocessing internally (deskew, binarize, denoise).
# This module is kept as a passthrough for interface compatibility.

from PIL import Image


def preprocess(image: Image.Image) -> Image.Image:
    return image
