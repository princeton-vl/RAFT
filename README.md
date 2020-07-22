# RAFT

**7/22/2020: We have updated our method to predict flow at full resolution leading to improved results on public benchmarks. This repository will be updated to reflect these changes within the next few days.**

This repository contains the source code for our paper:

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>
ECCV 2020 <br/>
Zachary Teed and Jia Deng<br/>

<img src="RAFT.png">

## Requirements
Our code was tested using PyTorch 1.3.1 and Python 3. The following additional packages need to be installed

  ```Shell
  pip install Pillow
  pip install scipy
  pip install opencv-python
  ```

## Demos
Pretrained models can be downloaded by running
```Shell
./scripts/download_models.sh
```

You can run the demos using one of the available models.

```Shell
python demo.py --model=models/chairs+things.pth
```

or using the small (1M parameter) model

```Shell
python demo.py --model=models/small.pth --small
```

Running the demos will display the two images and a vizualization of the optical flow estimate. After the images display, press any key to continue.

## Training
To train RAFT, you will need to download the required datasets. The first stage of training requires the [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs) and [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) datasets. Finetuning and evaluation require the [Sintel](http://sintel.is.tue.mpg.de/) and [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) datasets. We organize the directory structure as follows. By default `datasets.py` will search for the datasets in these locations

```Shell
├── datasets
│   ├── Sintel
|   |   ├── test
|   |   ├── training
│   ├── KITTI
|   |   ├── testing
|   |   ├── training
|   |   ├── devkit
│   ├── FlyingChairs_release
|   |   ├── data
│   ├── FlyingThings3D
|   |   ├── frames_cleanpass
|   |   ├── frames_finalpass
|   |   ├── optical_flow
```

We used the following training schedule in our paper (note: we use 2 GPUs for training)

```Shell
python train.py --name=chairs --image_size 368 496 --dataset=chairs --num_steps=100000 --lr=0.0002 --batch_size=6
```

Next, finetune on the FlyingThings dataset

```Shell
python train.py --name=things --image_size 368 768 --dataset=things --num_steps=60000 --lr=0.00005 --batch_size=3 --restore_ckpt=checkpoints/chairs.pth
```

You can perform dataset specific finetuning

### Sintel

```Shell
python train.py --name=sintel_ft --image_size 368 768 --dataset=sintel --num_steps=60000 --lr=0.00005 --batch_size=4 --restore_ckpt=checkpoints/things.pth
```

### KITTI

```Shell
python train.py --name=kitti_ft --image_size 288 896 --dataset=kitti --num_steps=40000 --lr=0.0001 --batch_size=4 --restore_ckpt=checkpoints/things.pth
```


## Evaluation
You can evaluate a model on Sintel and KITTI by running

```Shell
python evaluate.py --model=models/chairs+things.pth
```

or the small model by including the `small` flag

```Shell
python evaluate.py --model=models/small.pth --small
```
