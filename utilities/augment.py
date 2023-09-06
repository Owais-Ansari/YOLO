from  albumentations import *
import torch
import numpy as np
from albumentations.pytorch import ToTensorV2
import cv2
from skimage import color
#imgagenet mean and std
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

'''
From albumentations
    You import the required libraries.
    You define an augmentation pipeline.
    You read images and bounding boxes from the disk.
    You pass an image and bounding boxes to the augmentation pipeline and receive augmented images and boxes.

'''

class HEDJitter(object):
    # Psuedo Code
    # 1. Change from RGB --> HED space
    # 2. Jitter channels
    # 3. HED--> RGB
    # 4. Blend white region from original image

    # theta determines the amount of jittering per channel in HED --> hematoxylin eosin DAB
    # theta = 0.02 is the defualt

    # 'th' is threshold for blending white background regions from original image. 'th' is in teh range [0,1]
    # default 'th' is 0.9 which corresponds to 0.9*255

    def __init__(self, theta=0.04):
        self.theta = theta
        self.th = 0.9
        self.cutoff_range = [0.15, 0.85]

    def __call__(self, img):
        patch_mean = np.mean(a=img) / 255.0
        if ((patch_mean <= self.cutoff_range[0]) or (patch_mean >= self.cutoff_range[1])):
            return (img)
        self.alpha = torch.distributions.uniform.Uniform(1 - self.theta, 1 + self.theta).sample(
            [1, 3])  # np.random.uniform(1 - self.theta, 1 + self.theta, (1, 3))
        self.beta = torch.distributions.uniform.Uniform(-self.theta, self.theta).sample(
            [1, 3])  # np.random.uniform(-self.theta, self.theta, (1, 3))
        # print(self.beta)
        img = np.array(img)
        s = color.rgb2hed(img)
        ns = self.alpha * s + self.beta  # perturbations on HED color space
        nimg = color.hed2rgb(ns)
        rsimg = np.clip(a=nimg, a_min=0.0, a_max=1.0)
        rsimg = (255 * rsimg).astype('uint8')
        return rsimg


class val_aug_od(object):
    '''
    Input: a numpy image (dim,dim,channel) and a mask (dim,dim)
    return a torch tensor for image and mask
    '''
    def __init__(self, size = 512):
        self.size = size
        self.norm = Compose(
            [  
                LongestMaxSize(max_size=self.size),
                PadIfNeeded(min_height=self.size, min_width=self.size, border_mode=cv2.BORDER_CONSTANT),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
                bbox_params=BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
                )
    def __call__(self,image,bboxes):
        input_img = np.asarray(image)
        target = bboxes
        transformed = self.norm(image = input_img,bboxes = target)
        return transformed['image'], transformed['bboxes']

class train_aug_od(object):
    '''
    Input: a numpy image (dim,dim,channel) and a mask (dim,dim)
    return a torch tensor for image and mask
    '''
    def __init__(self,size=416,scale = 1.1,hed=0.05):
        #super().__init__()
        self.HED  = HEDJitter(hed)
        self.size  = size
        self.scale = scale
        self.scale_aug = Compose(
                        [
                        LongestMaxSize(max_size=int(self.size * self.scale)),
                        PadIfNeeded(
                            min_height=int(self.size * self.scale),
                            min_width=int(self.size * self.scale),
                            border_mode=cv2.BORDER_CONSTANT,
                                ),
                        RandomCrop(width=self.size, height=self.size),
                    OneOf(
                            [
                                ShiftScaleRotate(
                                rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                                ),
                            IAAAffine(shear=15, p=0.5, mode="constant"),
                            ],p=1.0
                        )], bbox_params = BboxParams(format="yolo", min_visibility=0.4, label_fields=[]))
    
        self.color = Compose([
                            RGBShift(),
                            Equalize(p=0.05),
                            HueSaturationValue(),
                            ColorJitter(),
                            RandomBrightnessContrast(),
                            ChannelShuffle(),
                              ],bbox_params = BboxParams(format="yolo", min_visibility=0.4, label_fields=[]))  
        
        self.geometric = Compose([
                        Flip(),
                        Sharpen(alpha=(0.25, 0.5), lightness=(0.5, 1.0)),
                        RandomRotate90(),
                        Transpose(),
                        #RandomScale([0.8, 1.2], 2),
                        #Rotate(limit=15, border_mode=cv2.BORDER_REFLECT),
                        Blur(),
                        GaussNoise()
                        ],bbox_params = BboxParams(format="yolo", min_visibility=0.4, label_fields=[]))
        
        self.norm = Compose([
                Normalize(mean=mean, std=std),
                ToTensorV2()],bbox_params = BboxParams(format="yolo", min_visibility=0.4, label_fields=[]))  
           
    def __call__(self,image,bboxes):
        input_img = np.asarray(image)
        target = bboxes
        input_img   = self.HED(input_img)
        transformed = self.scale_aug(image = input_img, bboxes = target)
        transformed = self.color(image = transformed['image'], bboxes =  transformed['bboxes'])
        transformed = self.geometric(image = transformed['image'],bboxes =  transformed['bboxes'])
        transformed = self.norm(image = transformed['image'],bboxes =  transformed['bboxes'])
        input_img,target = transformed['image'], transformed['bboxes']
        return input_img, target
    


                            



class test_aug(object):
    '''
    Input: a numpy image (dim,dim,channel) and a mask (dim,dim)
    return a torch tensor for image and mask
    '''
    def __init__(self,size = 512):
        self.size = size
        self.norm = Compose([
                Resize(self.size, self.size),
                Normalize(mean=mean, std=std),
                ToTensorV2()])
        
    def __call__(self,image):
        input_img = np.asarray(image)
        #input_mask = np.asarray(mask)
        transformed = self.norm(image = input_img)
        return transformed['image']#, transformed['mask']



   