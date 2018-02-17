import torchvision.transforms as transforms

import numpy as np

import PIL.Image as Image
from PIL.Image import Image as PILImage

def curl(img):
    output_pil = isinstance(img, PILImage)
    if output_pil:
        img = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3).astype(np.uint8)
        
    img = img / (255 if img.dtype == np.uint8 else 1)
    gx, gy = np.gradient(img, axis=(0, 1))
    gx, gy = gx.T, gy.T
    
    cr = (- gy[2]).T
    cg = (gx[2]).T
    cb = (gx[1] - gy[0]).T
    result = np.abs(np.dstack([cr, cg, cb]))
    return result if not output_pil else Image.fromarray((result * 255).astype('uint8'), 'RGB')


TRANSFORM_CURL = transforms.Lambda(lambd=curl)
NORMALIZE_CURL = transforms.Normalize(mean=[0.5209, 0.5241, 0.5343], std=[0.0289, 0.0309, 0.0417])
NORMALIZE = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))