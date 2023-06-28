import torch

from modules.frame_sampler.clip_saliency_frame_sampler.train import train_salient_frame_sampler
from modules.frame_sampler.clip_saliency_frame_sampler.model import SaliencyNet, CLIPTeacher

from torch.utils.data import DataLoader
from dataloaders.dataloader_activitynet_for_BCE import VideoDescriptionDataloader



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()

    student = SaliencyNet()

    optimizer = torch.optim.Adam(student.parameters(), lr=0.0001)

    epochs = 10

    train_json_file = "/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/train.json"
    train_frame_dir = "/media/external_10TB/10TB/p_haghighi/ActivityNet/sampled_frames/train"

    val_json_file = "/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/val_1.json"
    val_frame_dir = "/media/external_10TB/10TB/p_haghighi/ActivityNet/sampled_frames/val_1"


    gt_scores_file = "/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/CLIP-guided/CLIP-guided_video_summary_dataset_train.json"

    train_dataset = VideoDescriptionDataloader(train_json_file, train_frame_dir, gt_scores_file)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    val_dataset = VideoDescriptionDataloader(val_json_file, val_frame_dir, gt_scores_file)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)


    train_salient_frame_sampler(
        student, train_dataloader, val_dataloader, epochs, optimizer, device)


if __name__ == '__main__':
    main()
