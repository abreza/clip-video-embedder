import json
import cv2
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from utils.video_loader import download_video_from_youtube


json_path = './test_videodatainfo.json'

class MSRVTTDataset(Dataset):
    def __init__(self, path_to_videos, transform=None):
        self.transform = transform if transform else ToTensor()

        with open(json_path) as f:
            data = json.load(f)
        self.videos = []

        for video in data['videos']:
            try:
                video_path = path_to_videos+video['video_id']+'.mp4'

                #video_path = download_video_from_youtube(video['url'])

                self.videos.append({
                    "path": video_path,
                    "start_time": video['start time'],
                    "end_time": video['end time'],
                    "descriptions": [
                        item["caption"]
                        for item in data["sentences"]
                        if item["video_id"] == video["video_id"]
                    ]
                })
                self.descriptions.append(video['description'])
            except Exception as err:
                print(err)

    def __len__(self):
        return len(self.videos)

    def read_frames(self, video_path, start_time, end_time):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        frames = []

        for i in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(frame))

        cap.release()
        return frames

    def __getitem__(self, idx):
        video = self.videos[idx]
        frames = self.read_frames(
            video['path'], video["start_time"], video["end_time"])
        return frames, video['descriptions']
