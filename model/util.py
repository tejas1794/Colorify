from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as nnfunc


def load_image(path):
    """Loads image from a path and returns numpy array"""
    out_np = np.asarray(Image.open(path))
    if out_np.ndim == 2:
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np


def resize(img, dims=(256, 256), resample=3):
    """Handles actual resizing of array"""
    return np.asarray(Image.fromarray(img).resize((dims[1], dims[0]), resample=resample))


def preprocess(orig, dims=(256, 256), resample=3):
    """Handles pre-processing - converts image from rgb to lab and resizes to dims"""
    # Resize Image
    resized = resize(orig, dims=dims, resample=resample)
    # Convert resized and original images to LAB
    lab_orig = (color.rgb2lab(orig))[:, :, 0]
    lab_resized = (color.rgb2lab(resized))[:, :, 0]
    # Return tensors created out of images
    return torch.Tensor(lab_orig)[None, None, :, :], torch.Tensor(lab_resized)[None, None, :, :]


def postprocess(orig, output, mode='bilinear'):
    """Handles post-processing - converts image from lab to rgb and resizes"""
    # orig 	        1 x 1 x H_orig x W_orig
    # output 		1 x 2 x H x W

    dims_orig = orig.shape[2:]
    dims = output.shape[2:]
    # Resize if needed
    output_orig = output
    if dims_orig[0] != dims[0] or dims_orig[1] != dims[1]:
        output_orig = nnfunc.interpolate(output, size=dims_orig, mode='bilinear')
    # Convert to RGB from LAB space
    return color.lab2rgb(torch.cat((orig, output_orig), dim=1).data.cpu().numpy()[0, ...].transpose((1, 2, 0)))
