import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
from torchvision import transforms
import sys, os
from model import get_model
from data import base_instance, target_instance

xception_default_data_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transform
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image

fake_data_path = '/exports/eddie/scratch/s2017377/code/ff/FaceForensics/dataset/manipulated_sequences/DeepFakeDetection/c23/videos'
original_data_path_actors = '/exports/eddie/scratch/s2017377/code/ff/FaceForensics/dataset/original_sequences/actors/c23/videos'
original_data_path_youtube = '/exports/eddie/scratch/s2017377/code/ff/FaceForensics/dataset/original_sequences/youtube/c23/videos'

fake_data = os.listdir(fake_data_path)
original_data = os.listdir(original_data_path_actors) + os.listdir(original_data_path_youtube)

model = get_model()
device = torch.device("cuda:0")

# Try classifying fake as real
base_instance_vid = original_data[0]
target_instance_vid = fake_data[0]

base_instance_path = base_instance(base_instance_vid)
target_instance_path = target_instance(target_instance_vid)

max_iters = 100

def create_poison():
    print(model)
    feature_space = nn.Sequential(*list(model.model.children())[:-1])
    print(feature_space)
    feature_space = feature_space.to(device)
    for name, module in feature_space.named_modules():
        for param in module.parameters():
            param.requires_grad = True
    
    x = base_instance
    for _ in range(max_iters):
        x,loss = poison_iteration(feature_space, x.detach(), base_instance, target_instance)
    return None

def poison_iteration(feature_space, x, base_instance, target_instance, beta=0.25, lr=0.01):
    x.requires_grad = True
    
    feature_space.eval()

    fs_t = feature_space(target_instance.view(1,*target_instance.shape)).detach()
    fs_t.requires_grad = False
    
    # Forward Step:
    dif = feature_space(x.view(1,*x.shape))-fs_t
    loss = torch.sum(torch.mul(dif,dif))
    loss.backward()

    x2 = x.clone()
    x2-=(x.grad*lr)

    # Backward Step:
    x = (x2+lr*beta*base_instance)/(1+lr*beta)
    
    return x, loss.item()

poison_instance = create_poison()

def retrain_model():
    return None

new_model = retrain_model()

