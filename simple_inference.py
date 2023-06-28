import torch

from modules.frame_sampler.clip_saliency_frame_sampler.train import train_salient_frame_sampler
from modules.frame_sampler.clip_saliency_frame_sampler.model import SaliencyNet, CLIPTeacher

from torch.utils.data import DataLoader
from dataloaders.dataloader_activitynet_for_BCE import VideoDescriptionDataloader
import numpy as np

import matplotlib.pyplot as plt


def inference(sampler, dataloader, device ):
    i = 0
    for frames, gt_score in dataloader:
        frames = torch.squeeze(torch.tensor(
            np.stack(frames[0])), dim=1).to(device)
        
        saliency_scores = sampler(frames)

        # x = range(1,len(gt_score[0])+1)
        plt.figure()
        
        plt.plot(saliency_scores.detach().cpu().numpy(), label='saliency')
        plt.plot(gt_score[0], label='gt')
        plt.legend()

        plt.savefig('result' + str(i) + '.png')

        i+=1

        if i ==10: break



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()

    student = SaliencyNet()
    state_dict = torch.load('saved_model/student_model_epoch_10.pt')
    student.load_state_dict(state_dict)
    student.to(device)


    train_json_file = "/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/val_1.json"
    train_frame_dir = "/media/external_10TB/10TB/p_haghighi/ActivityNet/sampled_frames/val_1"

    gt_scores_file = "/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/CLIP-guided/CLIP-guided_video_summary_dataset_val_1.json"

    train_dataset = VideoDescriptionDataloader(train_json_file, train_frame_dir, gt_scores_file)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)


    inference(student, train_dataloader, device)

if __name__ == '__main__':
    main()
