import os
import pytube


def download_video_from_youtube(video_id, destination_path=None, video_name=None):
    destination_path = destination_path if destination_path else '/content/'
    video_name = video_name if video_name else f'{video_id}.mp4'
    print(f'Downloading YouTube video {video_id}.')
    pytube.YouTube(f'https://youtu.be/{video_id}').streams.get_highest_resolution().download(
        destination_path, filename=video_name)
    print(f'Download complete.')
    return os.path.join(destination_path, video_name)
