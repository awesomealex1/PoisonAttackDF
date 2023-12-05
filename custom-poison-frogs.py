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

model_path = '/exports/eddie/scratch/s2017377/code/ff/FaceForensics/classification/network/faceforensics++_models_subset/full/xception/full_c23.p'
fake_data_path = '/exports/eddie/scratch/s2017377/code/ff/FaceForensics/dataset/manipulated_sequences/DeepFakeDetection/c23/videos'
original_data_path_actors = '/exports/eddie/scratch/s2017377/code/ff/FaceForensics/dataset/original_sequences/actors/c23/videos'
original_data_path_youtube = '/exports/eddie/scratch/s2017377/code/ff/FaceForensics/dataset/original_sequences/youtube/c23/videos'

fake_data = os.listdir(fake_data_path)
original_data = os.listdir(original_data_path_actors) + os.listdir(original_data_path_youtube)

# Face detector
face_detector = dlib.get_frontal_face_detector()

cuda = True

# Load model
if model_path is not None:
    if not cuda:
        model = torch.load(model_path, map_location = "cpu")
    else:
        model = torch.load(model_path)
    print('Model found in {}'.format(model_path))
else:
    print('No model found, initializing random model.')
if cuda:
    print("Converting mode to cuda")
    model = model.cuda()
    for param in model.parameters():
        param.requires_grad = True
    print("Converted to cuda")

# Try classifying fake as real
base_instance = original_data[0]
target_instance = fake_data[0]

max_iters = 10

def create_poison():
    print(model)
    feature_space = nn.Sequential(*list(model.classifier.children())[:-2])

    for _ in range(max_iters):
        forward_step
        backward_step
    return None

poison_instance = create_poison()

def retrain_model():
    return None

new_model = retrain_model()

