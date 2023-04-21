import torch
from torch.utils.data import DataLoader

from datasets.msrvtt.dataset import MSRVTTDataset

from modules.frame_sampler.clip_saliency_frame_sampler.model import SaliencyNet, CLIPTeacher
from modules.frame_sampler.clip_saliency_frame_sampler.train import train_salient_frame_sampler


import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set_path', type=str,
                        required=True, help='Path to tarinset videos')
    parser.add_argument('--test_set_path', type=str,
                        required=True, help='Path to tarinset videos')
    args = parser.parse_args()
    return vars(args)


def main():
    args = get_args()
    train_set_path = args['train_set_path']
    test_set_path = args['test_set_path']

    msr_vtt_trainset = MSRVTTDataset(train_set_path)
    train_dataloader = DataLoader(msr_vtt_trainset, batch_size=1)

    msr_vtt_testset = MSRVTTDataset(test_set_path)
    val_dataloader = DataLoader(msr_vtt_testset, batch_size=1)

    saliency_net = SaliencyNet()
    clip_teacher = CLIPTeacher()

    optimizer = torch.optim.Adam(saliency_net.parameters(), lr=1e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_salient_frame_sampler(clip_teacher, saliency_net,
                                train_dataloader, val_dataloader,
                                epochs=10, optimizer=optimizer, device=device)


if __name__ == '__main__':
    main()
