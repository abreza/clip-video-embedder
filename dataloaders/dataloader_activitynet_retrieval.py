from torch.utils.data import Dataset
import json
from dataloaders.rawvideo_util import RawVideoExtractor

class ActivityNet_DataLoader(Dataset):
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            max_words=30,
            feature_framerate=1.0,
            image_resolution=224,
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words

        self.subset = subset
        assert self.subset in ["train", "val"]


        data_path = f'datasets/ActivityNet/{self.subset}.json'

        with open(data_path) as f:
            self.data = json.load(f)

        self.list_data = []

        for video_id in self.data:
            item = self.data[video_id]
            self.list_data.append({
                "youtube_video_id": video_id[2:],
                "segments": [
                    {"description": cap, "timestamp": time}
                    for cap, time in zip(item["sentences"], item["timestamps"])
                ]
            })

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, feature_idx):
        print('!!!!!!!!!!!!!!!')
        data = self.list_data[feature_idx]
        descriptions = [item['description'] for item in data['segments']]
        images = self.rawVideoExtractor.get_video_data(data['youtube_video_id'])
        print(images.shape, len(descriptions))
        return images, descriptions
