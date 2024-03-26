import torch
import glob
import numpy as np
import cv2
from collections import OrderedDict
import os
import math
import glob
import yaml
import scipy.io as sio
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import joblib
from PIL import Image
import random
from torchvision import transforms
import json

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

OBJECT_TYPE = [
# "candle",
# "capsules",
# "cashew",
# "chewinggum",
# "fryum",
# "macaroni1",
# "macaroni2",
# "pcb1",
# "pcb2",
# "pcb3",
# "pcb4",
# "pipe_fryum",
'wood',
'bottle',
'capsule',
'pill',
'transistor',
'zipper',
'cable',
'hazelnut',
'metal_nut',
'screw',
'toothbrush',
'all']

def get_inputs(file_addr, read_flag=None):
    file_format = file_addr.split('.')[-1]
    if file_format == 'mat':
        return sio.loadmat(file_addr, verify_compressed_data_integrity=False)['uv']
    elif file_format == 'npy':
        return np.load(file_addr)
    else:
        if read_flag is not None:
            img = cv2.imread(file_addr, read_flag)
        else:
            img = cv2.imread(file_addr)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


def img_tensor2numpy(img):
    # mutual transformation between ndarray-like imgs and Tensor-like images
    # both intensity and rgb images are represented by 3-dim data
    if isinstance(img, np.ndarray):
        return torch.from_numpy(np.transpose(img, [2, 0, 1]))
    else:
        return np.transpose(img, [1, 2, 0]).numpy()


def img_batch_tensor2numpy(img_batch):
    # both intensity and rgb image batch are represented by 4-dim data
    if isinstance(img_batch, np.ndarray):
        if len(img_batch.shape) == 4:
            return torch.from_numpy(np.transpose(img_batch, [0, 3, 1, 2]))
        else:
            return torch.from_numpy(np.transpose(img_batch, [0, 1, 4, 2, 3]))
    else:
        if len(img_batch.numpy().shape) == 4:
            return np.transpose(img_batch, [0, 2, 3, 1]).numpy()
        else:
            return np.transpose(img_batch, [0, 1, 3, 4, 2]).numpy()

def _convert_to_rgb(image):
    return image.convert('RGB')

class mvtec_dataset(Dataset):
    def __init__(self, config, data_dir, spatial_size=240, mode="train", shot=0, preprocess=None):
        super(mvtec_dataset, self).__init__()
        self.shot = shot
        self.data_dir = data_dir
        self.spatial_size = spatial_size
        self.mask_transform = transforms.ToTensor()
        self.pre_transform = transforms.Compose([
                transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size=(240, 240)),
                _convert_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        self.post_transform = transforms.Compose([
                transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        self.obj_type = config['obj_type']
        self.mode = mode
        self.preprocess = preprocess
        self.dataset_init()
        self.resize_shape = (spatial_size, spatial_size)
        

    def dataset_init(self):
        if self.shot == 'zero':
            self.img_paths = []
            return
        if self.obj_type == 'all':
            if self.mode == 'train':
                self.img_paths = glob.glob(os.path.join(self.data_dir, '**', 'train', 'good', '*.*'))
            else:
                self.img_paths = glob.glob(os.path.join(self.data_dir, '**', 'test', '**', '*.*'))
        else:
            type_dir = os.path.join(self.data_dir, self.obj_type)
            if self.mode == 'train':
                img_dir = os.path.join(type_dir, 'train', 'good')
                self.img_paths = glob.glob(os.path.join(img_dir, '*.*'))
            else:
                self.img_paths = glob.glob(os.path.join(type_dir, 'test', '**', '*.*'))
        self.img_paths = sorted(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def resize(self, sample):
        return np.array(cv2.resize(sample, (self.spatial_size, self.spatial_size)))

    def tile_image(self, img, stride_ratio=0.8):
        height, width = img.size
        shorter_edge = min(height, width)
        larger_edge = max(height, width)
        tile_size = shorter_edge
        stride = int(tile_size * stride_ratio)

        tile_image = []
        for x in range(0, larger_edge, stride):
            if x + tile_size >= larger_edge:
                tile = img.crop((larger_edge - tile_size, 0, larger_edge, tile_size))
            else:
                tile = img.crop((x, 0, x + tile_size, tile_size))
            tile_image.append(tile)
        return tile_image

    def transform_image(self, image_path):
        # load image
        image = Image.open(image_path)
        height, width = image.size

        if height == width:
            # pre-process
            preprocessed_image = self.pre_transform(image)
            transformed_image = preprocessed_image
            return transformed_image
        else:
            tile_image = self.tile_image(image, stride_ratio=0.8)
            transformed_image = []
            for img in tile_image:
                # pre-process
                preprocessed_image = self.pre_transform(img)
                trans_img = preprocessed_image
                transformed_image.append(trans_img)

            return transformed_image

    def __getitem__(self, indice):
        """
            returns:
                image: 3,256,256
                mask:  1,256,256
                has_anomaly: 1
                defect_type: 3
                dice: 10
        """
        normal_list = []

        if self.shot != 0:
            type_dir = os.path.join(self.data_dir, self.obj_type)
            img_dir = os.path.join(type_dir, 'train', 'good')
            all_ref_file = os.listdir(os.path.join(img_dir))
            m = np.random.RandomState(10).choice(all_ref_file, self.shot)

            # normal_json_path = "./2_shot.json"
            # with open(normal_json_path, 'r') as f:
            #     normal_json = json.load(f)
            # m = normal_json[self.obj_type]

            for file in m:
                tmp_dir = os.path.join(img_dir, file)
                normal_list.append(tmp_dir)

        img_path = self.img_paths[indice]

        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)

        if base_dir == 'good':
            has_anomaly = np.array([0], dtype=np.float32)
            mask = np.zeros((self.spatial_size, self.spatial_size)) / 255.0
        else:
            # mask_path = os.path.join(dir_path, '../../ground_truth/')
            # mask_path = os.path.join(mask_path, base_dir)
            # mask_file_name = file_name.split(".")[0]+"_mask.png"
            # mask_path = os.path.join(mask_path, mask_file_name)

            has_anomaly = np.array([1], dtype=np.float32)
            mask = np.zeros((self.spatial_size, self.spatial_size)) / 255.0
            # mask = get_inputs(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
        
        if self.preprocess is not None:
            image = self.preprocess(Image.open(img_path))
            ref_list = []
            for i in normal_list:
                ref = self.preprocess(Image.open(i))
                ref_list.append(ref)
        else:
            image = self.transform_image(img_path)
            ref_list = []
            for i in normal_list:
                ref = self.transform_image(i)
                ref_list.append(ref)


        mask = self.resize(mask)
        mask = self.mask_transform(mask)
        indice = np.array([indice], dtype=np.float32)

        return image, ref_list, mask, has_anomaly, indice