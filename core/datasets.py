# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import cv2
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, FlowAugmentorKITTI


class CombinedDataset(data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        length = 0 
        for i in range(len(self.datasets)):
            length += len(self.datsaets[i])
        return length

    def __getitem__(self, index):
        i = 0
        for j in range(len(self.datasets)):
            if i + len(self.datasets[j]) >= index:
                yield self.datasets[j][index-i]
                break
            i += len(self.datasets[j])

    def __add__(self, other):
        self.datasets.append(other)
        return self

class FlowDataset(data.Dataset):
    def __init__(self, args, image_size=None, do_augument=False):
        self.image_size = image_size
        self.do_augument = do_augument

        if self.do_augument:
            self.augumentor = FlowAugmentor(self.image_size)

        self.flow_list = []
        self.image_list = []

        self.init_seed = False

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        flow = frame_utils.read_gen(self.flow_list[index])
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)[..., :3]
        img2 = np.array(img2).astype(np.uint8)[..., :3]
        flow = np.array(flow).astype(np.float32)

        if self.do_augument:
            img1, img2, flow = self.augumentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        valid = torch.ones_like(flow[0])

        return img1, img2, flow, valid

    def __len__(self):
        return len(self.image_list)

    def __add(self, other):
        return CombinedDataset([self, other])


class MpiSintelTest(FlowDataset):
    def __init__(self, args, root='datasets/Sintel/test', dstype='clean'):
        super(MpiSintelTest, self).__init__(args, image_size=None, do_augument=False)

        self.root = root
        self.dstype = dstype

        image_dir = osp.join(self.root, dstype)
        all_sequences = os.listdir(image_dir)

        self.image_list = []
        for sequence in all_sequences:
            frames = sorted(glob(osp.join(image_dir, sequence, '*.png')))
            for i in range(len(frames)-1):
                self.image_list += [[frames[i], frames[i+1], sequence, i]]

    def __getitem__(self, index):
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        sequence = self.image_list[index][2]
        frame = self.image_list[index][3]

        img1 = np.array(img1).astype(np.uint8)[..., :3]
        img2 = np.array(img2).astype(np.uint8)[..., :3]
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        return img1, img2, sequence, frame


class MpiSintel(FlowDataset):
    def __init__(self, args, image_size=None, do_augument=True, root='datasets/Sintel/training', dstype='clean'):
        super(MpiSintel, self).__init__(args, image_size, do_augument)
        if do_augument:
            self.augumentor.min_scale = -0.2
            self.augumentor.max_scale = 0.7

        self.root = root
        self.dstype = dstype

        flow_root = osp.join(root, 'flow')
        image_root = osp.join(root, dstype)

        file_list = sorted(glob(osp.join(flow_root, '*/*.flo')))
        for flo in file_list:
            fbase = flo[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = osp.join(image_root, fprefix + "%04d"%(fnum+0) + '.png')
            img2 = osp.join(image_root, fprefix + "%04d"%(fnum+1) + '.png')

            if not osp.isfile(img1) or not osp.isfile(img2) or not osp.isfile(flo):
                continue

            self.image_list.append((img1, img2))
            self.flow_list.append(flo)


class FlyingChairs(FlowDataset):
    def __init__(self, args, image_size=None, do_augument=True, root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(args, image_size, do_augument)
        self.root = root
        self.augumentor.min_scale = -0.2
        self.augumentor.max_scale = 1.0

        images = sorted(glob(osp.join(root, '*.ppm')))
        self.flow_list = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(self.flow_list))

        self.image_list = []
        for i in range(len(self.flow_list)):
            im1 = images[2*i]
            im2 = images[2*i + 1]
            self.image_list.append([im1, im2])


class SceneFlow(FlowDataset):
    def __init__(self, args, image_size, do_augument=True, root='datasets',
            dstype='frames_cleanpass', use_flyingthings=True, use_monkaa=False, use_driving=False):
        
        super(SceneFlow, self).__init__(args, image_size, do_augument)
        self.root = root
        self.dstype = dstype

        self.augumentor.min_scale = -0.2
        self.augumentor.max_scale = 0.8

        if use_flyingthings:
            self.add_flyingthings()
        
        if use_monkaa:
            self.add_monkaa()

        if use_driving:
            self.add_driving()

    def add_flyingthings(self):
        root = osp.join(self.root, 'FlyingThings3D')

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, self.dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i+1]]]
                            self.flow_list += [flows[i]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i+1], images[i]]]
                            self.flow_list += [flows[i+1]]
      
    def add_monkaa(self):
        pass # we don't use monkaa

    def add_driving(self):
        pass # we don't use driving


class KITTI(FlowDataset):
    def __init__(self, args, image_size=None, do_augument=True, is_test=False, is_val=False, do_pad=False, split=True, root='datasets/KITTI'):
        super(KITTI, self).__init__(args, image_size, do_augument)
        self.root = root
        self.is_test = is_test
        self.is_val = is_val
        self.do_pad = do_pad

        if self.do_augument:
            self.augumentor = FlowAugumentorKITTI(self.image_size, args.eraser_aug, min_scale=-0.2, max_scale=0.5)

        if self.is_test:
            images1 = sorted(glob(os.path.join(root, 'testing', 'image_2/*_10.png')))
            images2 = sorted(glob(os.path.join(root, 'testing', 'image_2/*_11.png')))
            for i in range(len(images1)):
                self.image_list += [[images1[i], images2[i]]]

        else:
            flows = sorted(glob(os.path.join(root, 'training', 'flow_occ/*_10.png')))
            images1 = sorted(glob(os.path.join(root, 'training', 'image_2/*_10.png')))
            images2 = sorted(glob(os.path.join(root, 'training', 'image_2/*_11.png')))

            for i in range(len(flows)):
                self.flow_list += [flows[i]]
                self.image_list += [[images1[i], images2[i]]]


    def __getitem__(self, index):

        if self.is_test:
            frame_id = self.image_list[index][0]
            frame_id = frame_id.split('/')[-1]

            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])

            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, frame_id


        else:
            if not self.init_seed:
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is not None:
                    np.random.seed(worker_info.id)
                    random.seed(worker_info.id)
                    self.init_seed = True

            index = index % len(self.image_list)
            frame_id = self.image_list[index][0]
            frame_id = frame_id.split('/')[-1]

            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])

            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]

            if self.do_augument:
                img1, img2, flow, valid = self.augumentor(img1, img2, flow, valid)

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            valid = torch.from_numpy(valid).float()

            if self.do_pad:
                ht, wd = img1.shape[1:]
                pad_ht = (((ht // 8) + 1) * 8 - ht) % 8
                pad_wd = (((wd // 8) + 1) * 8 - wd) % 8
                pad_ht1 = [0, pad_ht]
                pad_wd1 = [pad_wd//2, pad_wd - pad_wd//2]
                pad = pad_wd1 + pad_ht1

                img1 = img1.view(1, 3, ht, wd)
                img2 = img2.view(1, 3, ht, wd)
                flow = flow.view(1, 2, ht, wd)
                valid = valid.view(1, 1, ht, wd)

                img1 = torch.nn.functional.pad(img1, pad, mode='replicate')
                img2 = torch.nn.functional.pad(img2, pad, mode='replicate')
                flow = torch.nn.functional.pad(flow, pad, mode='constant', value=0)
                valid = torch.nn.functional.pad(valid, pad, mode='replicate', value=0)

                img1 = img1.view(3, ht+pad_ht, wd+pad_wd)
                img2 = img2.view(3, ht+pad_ht, wd+pad_wd)
                flow = flow.view(2, ht+pad_ht, wd+pad_wd)
                valid = valid.view(ht+pad_ht, wd+pad_wd)

            if self.is_test:
                return img1, img2, flow, valid, frame_id

            return img1, img2, flow, valid
