import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class VideoDescriptionDataset(Dataset):
    def __init__(self, video_dir, json_file, frame_dir):
        self.video_dir = video_dir
        self.frame_dir = frame_dir
        with open(json_file, "r") as f:
            self.video_data = json.load(f)

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        # Load the textual description of the video
        video_id = list(self.video_data.keys())[idx]
        text_data = self.video_data[video_id]["sentences"]

        # Load the CLIP embeddings for the video
        clip_file = os.path.join(self.video_dir, f"{video_id}.pt")
        clip_embeddings = torch.load(clip_file)

        # Load the sampled frames for the video
        frame_dir = os.path.join(self.frame_dir, video_id)
        transform = transforms.ToTensor()
        sampled_frames = []
        for frame_file in os.listdir(frame_dir):
            frame_path = os.path.join(frame_dir, frame_file)
            frame = Image.open(frame_path)
            frame = transform(frame)
            sampled_frames.append(frame)
        sampled_frames = torch.stack(sampled_frames)

        # Return the three items as a tuple
        return text_data, clip_embeddings, sampled_frames

# Example usage
video_dir = "/media/10TB71/ramezani/ACTIVITY_NET/CLIP_embeddings"
json_file = "/path/to/video_descriptions.json"
frame_dir = "/media/10TB71/ramezani/ACTIVITY_NET/sampled_frames"
dataset = VideoDescriptionDataset(video_dir, json_file, frame_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for sentences, clip_embeddings, sampled_frames in dataloader:
    # Process the batch here