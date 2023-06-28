import os
import json
import torch
import random
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class VideoDescriptionDataset(Dataset):
    def __init__(self, embedding_dir, json_file, frame_dir, num_pairs_per_video=4):
        self.embedding_dir = embedding_dir
        self.frame_dir = frame_dir
        self.num_pairs_per_video = num_pairs_per_video
        with open(json_file, "r") as f:
            self.video_data = json.load(f)

    def __len__(self):
        return len(self.video_data) * self.num_pairs_per_video

    def __getitem__(self, idx):
        video_idx = idx // self.num_pairs_per_video
        video_id = list(self.video_data.keys())[video_idx]

        # Select two random frames from the video
        frame_dir = os.path.join(self.frame_dir, video_id)
        frame_files = os.listdir(frame_dir)
        pair = random.sample(frame_files, 2)
        transform = transforms.ToTensor()
        frames = [Image.open(os.path.join(frame_dir, frame)) for frame in pair]
        frames_tensor = [transform(f) for f in frames]

        return frames_tensor[0], frames_tensor[1]

class PairwiseFrameSampler:
    def __init__(self, dataset, num_videos=4):
        self.dataset = dataset
        self.num_videos = num_videos

    def __iter__(self):
        video_indices = list(range(len(self.dataset) // self.dataset.num_pairs_per_video))
        while True:
            selected_videos = random.sample(video_indices, self.num_videos)
            pairs = []
            for idx in selected_videos:
                pair_idx = idx * self.dataset.num_pairs_per_video + random.randint(0, self.dataset.num_pairs_per_video - 1)
                pairs.append(self.dataset[pair_idx])
            yield pairs


embedding_dir = "/media/10TB71/ramezani/ACTIVITY_NET/CLIP_embeddings"
json_file = "/path/to/video_descriptions.json"
frame_dir = "/media/10TB71/ramezani/ACTIVITY_NET/sampled_frames"
dataset = VideoDescriptionDataset(embedding_dir, json_file, frame_dir)
sampler = PairwiseFrameSampler(dataset)
dataloader = DataLoader(dataset, batch_sampler=sampler)