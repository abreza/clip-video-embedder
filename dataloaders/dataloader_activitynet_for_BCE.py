import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class VideoDescriptionDataloader(Dataset):
    def __init__(self, json_file, videos_dir, gt_scores_file_path):
        self.videos_dir = videos_dir
        with open(json_file, "r") as f:
            self.video_data = json.load(f)
                
        with open(gt_scores_file_path, "r") as f:
            self.gt_scores = json.load(f)

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        video_id = list(self.video_data.keys())[idx]

        frames_dir = os.path.join(self.videos_dir, video_id)
        transform = transforms.ToTensor()

        # Get the number of frames
        num_frames = len(os.listdir(frames_dir))

        # Create a placeholder tensor for all frames
        frames_tensor = torch.empty(num_frames, 3, 224, 224)  # Adjust the dimensions according to your frames

        # Iterate over frame files
        for i, frame_file_name in enumerate(os.listdir(frames_dir)):
            frame_path = os.path.join(frames_dir, frame_file_name)
            frame = Image.open(frame_path)
            frame = transform(frame)
            frames_tensor[i] = frame

        gt_score = torch.FloatTensor(self.gt_scores[video_id])

        min_value = gt_score.min()
        max_value = gt_score.max()

        normalized_data = (gt_score - min_value) / (max_value - min_value)

        return frames_tensor, normalized_data
