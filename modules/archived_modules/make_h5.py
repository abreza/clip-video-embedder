from torchvision import models, transforms
import torch.nn as nn
import torch
from PIL import Image
import os
import h5py
import json
from tqdm import tqdm

import sys
import gc
import time

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device =  torch.device("cpu")

model = models.googlenet(weights='IMAGENET1K_V1').to(device)
lenet = nn.Sequential(*list(model.children())[:-2])

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


gt_scores_file = "/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/CLIP-guided/CLIP-guided_video_summary_dataset_train.json"
sampled_frame_path = "/media/external_10TB/10TB/p_haghighi/ActivityNet/sampled_frames/train"


output_file = "output_train_gpu.h5"

with open(gt_scores_file, "r") as t:
    gt_scores = json.load(t)

with h5py.File(output_file, "w") as f:

    # Loop over the videos in the sampled frame path
    print(f"Processing videos of train")
    
    for video_id in tqdm(gt_scores.keys(), desc="Processing videos"):
        torch.cuda.empty_cache()

        video_path = os.path.join(sampled_frame_path, video_id)
        if not os.path.isdir(video_path): continue

        images = []

        for filename in sorted(os.listdir(video_path)):
            filepath = os.path.join(video_path, filename)
            image = Image.open(filepath)
            image = transform(image)
            images.append(image)

        images = torch.stack(images).to(device)

        with torch.no_grad():
            features = lenet(images).squeeze().detach().cpu().numpy()

        video_gt_scores = gt_scores.get(video_id, [])

        group = f.create_group(video_id)

        group.create_dataset("features", data=features)
        group.create_dataset("gtscore", data=video_gt_scores)
