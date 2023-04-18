import torch
from torch.utils.data import DataLoader

from datasets.msrvtt.dataset import MSRVTTDataset

from modules.frame_sampler.clip_saliency_frame_sampler.model import SaliencyNet, CLIPTeacher
from modules.frame_sampler.clip_saliency_frame_sampler.train import train_salient_frame_sampler


import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True, help='Path to videos')
    args = parser.parse_args()
    return vars(args)



def main():
    args = get_args()
    path_to_videos = args['video-path']

    msr_vtt_dataset = MSRVTTDataset(path_to_videos)
    dataloader = DataLoader(msr_vtt_dataset, batch_size=1)

    saliency_net = SaliencyNet()
    clip_teacher = CLIPTeacher()

    optimizer = torch.optim.Adam(saliency_net.parameters(), lr=1e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_salient_frame_sampler(clip_teacher, saliency_net,
                                dataloader, epochs=10, optimizer=optimizer, device=device)


if __name__ == '__main__':
    main()