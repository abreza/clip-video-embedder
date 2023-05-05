import os
import pytube


def download_video_from_youtube(video_id_or_url, destination_path=None, video_name=None, retry_count=3):
    if retry_count == 0:
        raise Exception('Can not download video!')
    try:
        video_url = video_id_or_url if "://" in video_id_or_url else f'https://youtu.be/{video_id_or_url}'

        video_id = pytube.extract.video_id(video_url)
        destination_path = destination_path if destination_path else '/content/'
        video_name = video_name if video_name else f'{video_id}.mp4'
        video_path = os.path.join(destination_path, video_name)

        if os.path.isfile(video_path):
            return video_path

        print(f'Downloading YouTube video {video_id}.')
        pytube.YouTube(video_url).streams.get_highest_resolution().download(
            destination_path, filename=video_name)
        print(f'Download complete.')

        return video_path
    except:
        download_video_from_youtube(
            video_id_or_url, destination_path, video_name, retry_count-1)
