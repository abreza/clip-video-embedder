import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader


def train_salient_frame_sampler(teacher, student, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int, optimizer, device):
    teacher.eval()
    teacher.to(device)
    student.train()
    student.to(device)

    criterion = nn.MSELoss()

    num_samples = 10

    for epoch in range(epochs):
        running_loss = 0.0
        for frames, descriptions in train_dataloader:
            frames = torch.squeeze(torch.tensor(np.stack(frames)), dim=1).to(device)
            descriptions = [desc[0] for desc in descriptions]

            with torch.no_grad():
                teacher_scores = teacher(frames, descriptions)

            student_scores = student(frames)

            optimizer.zero_grad()
            loss = criterion(student_scores, teacher_scores)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}")

        # Calculate validation loss
        with torch.no_grad():
            running_val_loss = 0.0
            for val_frames, val_descriptions in val_dataloader:
                val_frames = val_frames.to(device)
                val_descriptions = val_descriptions.to(device)

                val_student_scores = student(val_frames).squeeze()
                val_teacher_scores = teacher(val_frames, val_descriptions)

                val_loss = criterion(val_student_scores, val_teacher_scores)
                running_val_loss += val_loss.item()

            val_loss = running_val_loss / len(val_dataloader)
            print(
                f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")
