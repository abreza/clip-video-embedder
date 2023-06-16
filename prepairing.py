import os
import functools
import multiprocessing
from dataloaders.rawvideo_util import RawVideoExtractor

# import warnings
# warnings.filterwarnings("ignore", message="Argument 'interpolation' of type int is deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.")

framerate = 1

videos_path = '/media/external_10TB/10TB/p_haghighi/videos/'
sampled_frames_path = '/media/external_10TB/10TB/p_haghighi/sampled_frames/'

dataset_paths = [
    "v1-2/train",
    "v1-2/test",
    "v1-2/val",
    "v1-3/test",
    "v1-3/train_val"
]

def process_video(video_args):
    video_path, output_subfolder = video_args

    video_name = os.path.basename(video_path)
    video_id = os.path.splitext(video_name)[0] 
        
    if os.path.exists(output_subfolder) and os.listdir(output_subfolder):
        # print(f'{video_id} has been sampled!')
        return
        
    os.makedirs(output_subfolder, exist_ok=True)
        
    video_extractor = RawVideoExtractor(framerate=framerate, to_tensor=False)
    images = video_extractor.get_video_data(video_path)
        
    for i, img in enumerate(images):
        image_path = os.path.join(output_subfolder, f'{video_id}_frame{i:03d}.jpg')
        img.save(image_path)

    print(f'{dataset_path}/{video_name} --> {len(images):3d} sampled_frames')

def process_video_chunks(dataset_path):
    os.makedirs(sampled_frames_path+dataset_path, exist_ok=True)

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    video_list = os.listdir(videos_path+dataset_path)
    video_paths = [os.path.join(videos_path+dataset_path, video_name) for video_name in video_list]
    output_subfolders = [os.path.join(sampled_frames_path+dataset_path, os.path.splitext(video_name)[0]) for video_name in video_list]
    video_args = list(zip(video_paths, output_subfolders))
    pool.map(process_video, video_args)


for dataset_path in dataset_paths:
    print(f"Sampling of videos of {dataset_path} started!")
    process_video_chunks(dataset_path)
    print(f"Sampling of videos of {dataset_path} ended!")