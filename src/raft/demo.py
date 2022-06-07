import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from .core.utils import flow_viz
from . import inference



def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def demo(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = inference.load_model(
        args,
        device,
        args.model)

    def log(x):
        print(x)
        return x
        
    stream = (
        np.array(Image.open(log(impath))).astype(np.uint8)
        for impath
        in sorted(
            glob.glob(os.path.join(args.path, '*.png')) + 
            glob.glob(os.path.join(args.path, '*.jpg'))
        )
    )

    for image1, image2, flow_low, flow_up in inference.process_stream(stream, model, device, iters=20):
        viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="restore checkpoint")
    parser.add_argument('--path', required=True, help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
