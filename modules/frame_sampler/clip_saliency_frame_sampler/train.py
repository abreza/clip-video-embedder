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

    print_every = 50

    train_losses = []
    val_losses = []

    update_every = 5

    for epoch in range(epochs):
        epoch_loss = 0
        optimizer.zero_grad()

        for i, (frames, descriptions) in enumerate(train_dataloader):
            if len(frames) == 0:
                continue

            running_loss = 0.0
            frames = torch.squeeze(torch.tensor(
                np.stack(frames)), dim=1).to(device)
            descriptions = [desc[0] for desc in descriptions]

            with torch.no_grad():
                teacher_scores = teacher(frames, descriptions)

            student_scores = student(frames)

            loss = criterion(student_scores, teacher_scores)
            loss.backward()

            calculated_loss = loss.item()
            running_loss += calculated_loss
            epoch_loss += calculated_loss

            if (i + 1) % update_every == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (i + 1) % print_every == 0:
                print(
                    f'Epoch: {epoch + 1}, Batch: {i + 1}, Avg. Loss: {running_loss / print_every}')
                running_loss = 0.0

        # if the number of batches is not a multiple of 'update_every', update parameters for the remaining batches
        if len(train_dataloader) % update_every != 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}")

        torch.save(student.state_dict(), f'student_model_epoch_{epoch+1}.pt')

        with torch.no_grad():
            running_val_loss = 0.0
            for i, (val_frames, val_descriptions) in enumerate(val_dataloader):
                val_frames = torch.squeeze(torch.tensor(
                    np.stack(val_frames)), dim=1).to(device)
                val_descriptions = [desc[0] for desc in val_descriptions]

                val_teacher_scores = teacher(val_frames, val_descriptions)
                val_student_scores = student(val_frames)

                val_loss = criterion(val_student_scores, val_teacher_scores)
                running_val_loss += val_loss.item()

                if (i + 1) % print_every == 0:
                    print(
                        f'Validation, Batch: {i + 1}, Avg. Loss: {running_val_loss / print_every}')
                    running_val_loss = 0.0

            val_loss = running_val_loss / len(val_dataloader)
            val_losses.append(val_loss)
            print(
                f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")

    return train_losses, val_losses
