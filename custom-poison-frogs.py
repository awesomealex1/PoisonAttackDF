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
from model import get_model, predict_with_model
from data import base_instance, target_instance, get_boundingbox

xception_default_data_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

fake_data_path = '/exports/eddie/scratch/s2017377/code/ff/FaceForensics/dataset/manipulated_sequences/DeepFakeDetection/c23/videos'
original_data_path_actors = '/exports/eddie/scratch/s2017377/code/ff/FaceForensics/dataset/original_sequences/actors/c23/videos'
original_data_path_youtube = '/exports/eddie/scratch/s2017377/code/ff/FaceForensics/dataset/original_sequences/youtube/c23/videos'

fake_data = os.listdir(fake_data_path)
original_data = os.listdir(original_data_path_actors)

face_detector = dlib.get_frontal_face_detector()
model = get_model()
device = torch.device("cuda:0")

# Try classifying fake as real
base_instance_vid = os.path.join(original_data_path_actors, original_data[0])
target_instance_vid = os.path.join(fake_data_path, fake_data[0])

base_instance_img = base_instance(base_instance_vid)
target_instance_img = target_instance(target_instance_vid)

base_height, base_width = base_instance_img.shape[:2]
target_height, target_width = target_instance_img.shape[:2]

base_instance_gray = cv2.cvtColor(base_instance_img, cv2.COLOR_BGR2GRAY)
target_instance_gray = cv2.cvtColor(target_instance_img, cv2.COLOR_BGR2GRAY)

base_faces = face_detector(base_instance_gray, 1)
target_faces = face_detector(target_instance_gray, 1)

if len(base_faces) == 0 or len(target_faces) == 0:
    print("No faces detected")

base_face = base_faces[0]
target_face = target_faces[0]

base_x, base_y, base_size = get_boundingbox(base_face, base_width, base_height)
target_x, target_y, target_size = get_boundingbox(target_face, target_width, target_height)

base_cropped = base_instance_img[base_y:base_y+base_size, base_x:base_x+base_size]
target_cropped = target_instance_img[target_y:target_y+target_size, target_x:target_x+target_size]

prediction, output = predict_with_model(base_cropped, model)

def create_poison():
    print(model)
    feature_space = nn.Sequential(*list(model.model.children())[:-1])
    print(feature_space)
    feature_space = feature_space.to(device)
    for name, module in feature_space.named_modules():
        for param in module.parameters():
            param.requires_grad = True
    
    x = base_instance
    max_iters = 100
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

