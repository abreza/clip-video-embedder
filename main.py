import torch

from modules.frame_sampler.clip_saliency_frame_sampler.train import train_salient_frame_sampler
from modules.frame_sampler.clip_saliency_frame_sampler.model import SaliencyNet, CLIPTeacher

from utils.dotdict import DotDict
from dataloaders.data_dataloaders import dataloader_msvd_train


def main():

    args = DotDict(dict(data_path='/content/clip-video-embedder/datasets/MSVD/data.json',
                        features_path='/content/MSVD',
                        max_words=77,
                        feature_framerate=1,
                        max_frames=100,
                        train_frame_order=0,
                        slice_framepos=2,
                        batch_size=1,
                        n_gpu=1,
                        num_thread_reader=0))

    train_dataloader, train_length = dataloader_msvd_train(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()

    teacher = CLIPTeacher(device)
    student = SaliencyNet()

    optimizer = torch.optim.Adam(student.parameters(), lr=0.0001)

    val_dataloader = train_dataloader
    epochs = 1

    train_salient_frame_sampler(
        teacher, student, train_dataloader, val_dataloader, epochs, optimizer, device)


if __name__ == '__main__':
    main()
