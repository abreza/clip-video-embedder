import torch


def uniform_sample_frames(frames, num_samples):
    total_frames = frames.shape[0]
    step_size = total_frames // num_samples
    indices = torch.arange(0, total_frames, step_size)[:num_samples]
    return frames[indices]
