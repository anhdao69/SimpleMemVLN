"""Lossless current-frame transport shared by the two Python environments."""
import base64
import io
import numpy as np
from PIL import Image

SHAPE = (480, 640, 3)


def encode_rgb(rgb):
    rgb = np.asarray(rgb)
    if rgb.shape != SHAPE or rgb.dtype != np.uint8:
        raise ValueError(f'Expected uint8 RGB {SHAPE}, got {rgb.shape} {rgb.dtype}')
    return {'rgb_raw': base64.b64encode(rgb.tobytes()).decode('ascii')}


def decode_rgb(request):
    if 'rgb_raw' in request:
        raw = base64.b64decode(request['rgb_raw'], validate=True)
        if len(raw) != 480 * 640 * 3:
            raise ValueError('Invalid raw RGB byte count')
        return Image.frombytes('RGB', (640, 480), raw)
    with Image.open(io.BytesIO(base64.b64decode(request['rgb_png'], validate=True))) as image:
        if image.mode != 'RGB' or image.size != (640, 480):
            raise ValueError('Unexpected camera contract')
        return image.copy()
