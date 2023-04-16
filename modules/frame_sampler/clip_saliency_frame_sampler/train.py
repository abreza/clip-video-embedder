import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import CLIPTeacher, SaliencyNet


def train_salient_frame_sampler(teacher, student, dataloader: DataLoader, epochs: int, optimizer, device):
    teacher.eval()
    teacher.to(device)
    student.train()
    student.to(device)

    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for frames, descriptions in dataloader:
            frames = torch.stack(frames).to(device)

            student_scores = student(frames).squeeze()

            with torch.no_grad():
                teacher_scores = teacher(frames, descriptions)

            optimizer.zero_grad()
            loss = criterion(student_scores, teacher_scores)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
