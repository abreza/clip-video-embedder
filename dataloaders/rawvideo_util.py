import torch as th
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import cv2


class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            # Normalize((0.48145466, 0.4578275, 0.40821073),
            #           (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        cap = cv2.VideoCapture(video_file)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        start_frame = int(start_time * video_fps) if start_time else 0
        end_frame = int(end_time * video_fps) if end_time else frame_count - 1

        target_fps = sample_fp if sample_fp > 0 else video_fps
        target_interval = int(round(video_fps / target_fps))
        if target_interval == 0:
            target_interval = 1

        images = []

        # loop through each target frame index
        for i in range(start_frame, end_frame + 1):
            # set the current frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if ret:
                # get the timestamp of the current frame in milliseconds
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                # check whether the current frame should be sampled based on the target interval
                if (i - start_frame) % target_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    images.append(Image.fromarray(frame_rgb))
            else:
                break

        cap.release()

        return {'frames': images}

    def get_video_data(self, video_path, start_time=None, end_time=None):
        image_input = self.video_to_tensor(
            video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        return image_input

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1,
                                     tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data


# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2
