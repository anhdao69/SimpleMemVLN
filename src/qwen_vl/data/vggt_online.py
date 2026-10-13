"""CPU-only online teacher image staging; no teacher model in loader workers."""
import numpy as np
from PIL import Image
import torch

PREPROCESS_VERSION='omega-balanced512-rgb640x480-bicubic448x592-v1'
POSITION_VERSION='g1-center-sincos2048-rms1-scale0.5-v1'


def preprocess_teacher_images(paths):
    images=[]
    for path in paths:
        with Image.open(path) as image:
            if image.size!=(640,480) or image.mode!='RGB':
                raise ValueError('Online G1 expects dataset RGB640x480')
            array=np.array(image.resize((592,448),Image.Resampling.BICUBIC),copy=True)
        images.append(torch.from_numpy(array).permute(2,0,1).float().div(255))
    return torch.stack(images) if images else torch.empty(0,3,448,592)


def positional_grid():
    yy,xx=torch.meshgrid((torch.arange(15,dtype=torch.float32)+.5)/15,
                        (torch.arange(20,dtype=torch.float32)+.5)/20,indexing='ij')
    frequencies=10000**(-torch.arange(512,dtype=torch.float32)/512)
    a,b=(2*torch.pi*v.reshape(-1,1)*frequencies for v in (yy,xx))
    e=torch.cat((a.sin(),a.cos(),b.sin(),b.cos()),-1)
    return e/e.square().mean().sqrt()
