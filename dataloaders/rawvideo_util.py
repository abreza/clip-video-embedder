import numpy as np
from PIL import Image
import cv2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, framerate=-1 size=224 ):
        self.centercrop = centercrop
        self.framerate = framerate
        self.transform = self._transform(size)

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            # lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        cap = cv2.VideoCapture(video_file)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        start_frame = int(start_time * video_fps) if start_time else  0
        end_frame = int(end_time * video_fps) if end_time else frame_count - 1

        interval = 1
        if sample_fp > 0:
            interval = video_fps / sample_fp
        else:
            sample_fp = video_fps
        if interval == 0:
            interval = 1

        images = []

        for i in range(frame_count):
            ret, frame = cap.read()

            if ret:
                if i >= start_frame and i <= end_frame:
                    if len(images) * interval < i - start_frame:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        images.append(self.transform(Image.fromarray(frame_rgb)))

            else: 
                break

        cap.release()

        return images

    def get_video_data(self, video_path, start_time=None, end_time=None):
        image_input = self.video_to_tensor(
            video_path, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
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
