import torch

from .core.utils.utils import InputPadder
from .core.raft import RAFT

def preprocess(image, device):
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    image = image.unsqueeze(0)
    image = image.to(device)
    return image


def process_stream(stream, model, device, iters: int = 20):
    """
    Processes an image stream and generates tuples of (image1, image2, flow_low, flow_up)
    """
    it = iter(stream)
    image1 = next(it)
    image1 = preprocess(image1, device)

    model.eval()
    with torch.no_grad():
        for image2 in it:
            # preprocessing
            image2 = preprocess(image2, device)

            # pad so shapes match
            padder = InputPadder(image1.shape)
            image1p, image2p = padder.pad(image1, image2)

            # predict the flow
            flow_low, flow_up = model(image1p, image2p, iters=iters, test_mode=True)
            yield image1p, image2p, flow_low, flow_up

            image1 = image2


def cap_stream(cap, n: int = None):
    """
    Create an iterable of images from an OpenCV video capture object.
    :param n: Maximum number of frames to capture. None means unlimited.
    """
    frame_idx = 0
    while True:
        if n is not None and frame_idx >= n:
            break
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
        frame_idx += 1
        

def load_model(
    raft_args,
    device: torch.DeviceObjType,
    checkpoint_path: str):
    model = RAFT(raft_args)
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)
    pretrained_weights = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(pretrained_weights)
    return model.to(device)
