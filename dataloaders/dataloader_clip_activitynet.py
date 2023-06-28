import torch
from torch.utils.data import Dataset
import json

class ActivityNet_DataLoader(Dataset):
    def __init__(self, embeddings_directory, subset, data_path, max_frames=30):
        self.data_path = data_path
        self.max_frames = max_frames

        self.subset = subset
        assert self.subset in ["train", "test"]

        with open(data_path) as f:
            self.data = json.load(f)

        self.list_data = []
        self.frame_batches = []

        for video_id in self.data:
            item = self.data[video_id]
            self.list_data.append({
                "video_id": video_id,
                "segments": [
                    {"description": cap, "timestamp": time}
                    for cap, time in zip(item["sentences"], item["timestamps"])
                ]
            })
            frames = torch.load(f"{embeddings_directory}/{video_id}.pt")
            frame_batches_with_desc = self.split_and_pad_frames(frames, item["sentences"])
            self.frame_batches += frame_batches_with_desc

    def __len__(self):
        return len(self.frame_batches)

    def __getitem__(self, idx):
        return self.frame_batches[idx]

    def split_and_pad_frames(self, frames, descriptions):
        frame_batches = []
        total_frames = len(frames)

        for i in range(0, total_frames, self.max_frames):
            segment = frames[i:i+self.max_frames]
            segment_descriptions = descriptions[i//self.max_frames : (i+self.max_frames)//self.max_frames]

            if len(segment) < self.max_frames:
                pad_size = self.max_frames - len(segment)
                pad = torch.zeros((pad_size, *segment.shape[1:]))  # Assumes frames shape is (N, C, H, W)
                segment = torch.cat((segment, pad), dim=0)

            frame_batches.append((segment, segment_descriptions))

        return frame_batches