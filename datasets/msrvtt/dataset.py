import os
import json
import cv2
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from utils.video_loader import download_video_from_youtube


current_dir = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(current_dir, 'test_videodatainfo.json')
train_data_path = os.path.join(current_dir, 'train_val_videodatainfo.json')


class MSRVTTDataset(Dataset):
    def __init__(self, data_type='train', videos_path=os.path.join(current_dir, 'videos'), transform=None, max_size=None):
        self.transform = transform if transform else ToTensor()

        data_path = test_data_path if data_type == 'test' else train_data_path
        with open(data_path) as f:
            data = json.load(f)
        self.videos = []

        for video in data['videos']:
            if max_size and len(self.videos) >= max_size:
                break
            try:
                video_path = download_video_from_youtube(
                    video['url'], videos_path, f"{video['video_id']}.mp4")

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
            except Exception as err:
                print(err)

    def __len__(self):
        return len(self.videos)

    def read_frames(self, video_path, start_time, end_time, frame_per_second=2):
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        # sample one frame every `sample_rate` frames
        sample_rate = round(fps) // frame_per_second

        start_frame, end_frame = map(int, (start_time * fps, end_time * fps))

        sampled_frames = []

        i = -1

        while (cap.isOpened()):
            i += 1

            if i < start_frame or i > end_frame:
                continue

            ret, frame = cap.read()
            if ret == False:
                break

            # only save one frame every sample_rate frames
            if i % sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled_frames.append(self.transform(frame))

        cap.release()
        cv2.destroyAllWindows()

        return sampled_frames

    def __getitem__(self, idx):
        video = self.videos[idx]
        frames = self.read_frames(
            video['path'], video["start_time"], video["end_time"])
        return frames, video['descriptions']
