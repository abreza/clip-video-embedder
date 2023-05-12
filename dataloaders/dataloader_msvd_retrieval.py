import os
import json

from torch.utils.data import Dataset
from dataloaders.rawvideo_util import RawVideoExtractor


class MSVD_DataLoader(Dataset):
    """MSVD dataset loader."""

    def __init__(
            self,
            subset,
            data_path,
            features_path,
            max_words=30,
            feature_framerate=10.0,
            image_resolution=224,
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words

        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        video_id_path_dict = {}

        data_dir_path = os.path.dirname(data_path)
        video_id_path_dict["train"] = os.path.join(
            data_dir_path, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(
            data_dir_path, "val_list.txt")
        video_id_path_dict["test"] = os.path.join(
            data_dir_path, "test_list.txt")

        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip()[:11] for itm in fp.readlines()]

        with open(data_path) as f:
            self.data = json.load(f)

        self.list_data = []

        for item in self.data:
            if item['link'] not in video_ids:
                continue
            self.list_data.append({
                "youtube_video_id": item['link'],
                "segments": [
                    {"description": cap, "timestamp": [
                        item['start'], item['end']]}
                    for cap in item["description"]
                ]
            })

        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate, size=image_resolution)

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, feature_idx):
        data = self.list_data[feature_idx]
        [start, end] = data['segments'][0]['timestamp']
        descriptions = [item['description'] for item in data['segments']]
        video_path = f"{self.features_path}/{data['youtube_video_id']}_{start}_{end}.avi"
        images = self.rawVideoExtractor.get_video_data(video_path)
        return images, descriptions
