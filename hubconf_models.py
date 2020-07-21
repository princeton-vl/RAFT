import argparse
import io
import os
import time
import urllib.request
import zipfile
import torch
from torch.nn import functional as F

from .core.raft import RAFT as RAFT_module

models_url = "https://www.dropbox.com/s/a2acvmczgzm6f9n/models.zip?dl=1"  # dl=1 is important


__all__ = ["RAFT"]


ENV_TORCH_HOME = "TORCH_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"


def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            ENV_TORCH_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "torch")
        )
    )
    return torch_home


def _pad8(img):
    """pad image such that dimensions are divisible by 8"""
    ht, wd = img.shape[2:]
    pad_ht = (((ht // 8) + 1) * 8 - ht) % 8
    pad_wd = (((wd // 8) + 1) * 8 - wd) % 8
    pad_ht1 = [pad_ht // 2, pad_ht - pad_ht // 2]
    pad_wd1 = [pad_wd // 2, pad_wd - pad_wd // 2]

    img = F.pad(img, pad_wd1 + pad_ht1, mode="replicate")
    return img


def RAFT(pretrained=False, model_name="chairs+things", **kwargs):
    """
    RAFT model (https://arxiv.org/abs/2003.12039)
    model_name (str): One of 'chairs+things', 'sintel', 'kitti' and 'small'
                      note that for 'small', the architecture is smaller
    """

    model_list = ["chairs+things", "sintel", "kitti", "small"]
    if model_name not in model_list:
        raise ValueError("Model should be one of " + str(model_list))

    model_args = argparse.Namespace(**kwargs)
    model_args.small = "small" in model_name

    model = RAFT_module(model_args)
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = torch.nn.DataParallel(model, device_ids=[device])

    if pretrained:
        torch_home = _get_torch_home()
        model_dir = os.path.join(torch_home, "checkpoints", "models_RAFT")
        model_path = os.path.join(model_dir, "models", model_name + ".pth")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            response = urllib.request.urlopen(models_url, timeout=10)
            z = zipfile.ZipFile(io.BytesIO(response.read()))
            z.extractall(model_dir)
        else:
            time.sleep(10)  # Give the time for the models to be downloaded and unzipped

        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=map_location))

    model.to(device)
    model.eval()
    return model


def apply_model(model, images_from, images_to, iters=12, upsample=True):
    """
    Applies optical flow model to the pairs of images
    Args:
    images_from: torch.Tensor of size [B, H, W, C] containing RGB data for
        images that serve as optical flow source images
    images_to: torch.Tensor of size [B, H, W, C] containing RGB data for
        images that serve as optical flow destination images
    Return:
    optical_flow: torch.Tensor of size [B, H, W, 2]
    """
    images_from, images_to = _pad8(images_from), _pad8(images_to)
    with torch.no_grad():
        return model(image1=images_from, image2=images_to, iters=iters, upsample=upsample)
