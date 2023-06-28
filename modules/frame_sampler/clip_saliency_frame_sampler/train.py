import logging

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    filename='training_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def train_salient_frame_sampler(student, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int, optimizer, device):
    student.train()
    student.to(device)

    criterion = nn.MSELoss().to(device)

    print_every = 50

    epoch_train_losses = []
    epoch_val_losses = []

    update_every = 10

    for epoch in tqdm(range(epochs), desc="Epochs"):
        epoch_train_loss = 0
        print_loss = 0
        optimizer.zero_grad()

        for i, (frames, gt_score) in tqdm(enumerate(train_dataloader), desc="Training Batches", leave=False):
            
            if len(frames) == 0:
                print(f'video {i} has not any loaded frame')
                continue

            frames = torch.squeeze(torch.tensor(
                np.stack(frames[0])), dim=1).to(device)
                        
            student_scores = student(frames)

            gt_score = gt_score[0].to(device)
            loss = criterion(student_scores, gt_score)

            loss = loss / update_every

            loss.backward()

            calculated_loss = loss.item()
            print_loss += calculated_loss
            epoch_train_loss += calculated_loss

            if (i + 1) % update_every == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (i + 1) % print_every == 0:
                print(
                    f'Epoch: {epoch + 1}, Batch: {i + 1}, Avg. Loss: {print_loss / print_every}')
                logging.info(
                    f'Epoch: {epoch + 1}, Batch: {i + 1}, Avg. Loss: {print_loss / print_every}')
                print_loss = 0


        # if the number of batches is not a multiple of 'update_every', update parameters for the remaining batches
        if len(train_dataloader) % update_every != 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_train_loss = epoch_train_loss / len(train_dataloader)
        epoch_train_losses.append(epoch_train_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_train_loss:.5f}")
        logging.info(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_train_loss:.5f}")

        torch.save(student.state_dict(), f'student_model_epoch_{epoch+1}.pt')

        with torch.no_grad():
            running_val_loss = 0.0
            for i, (frames, gt_score) in tqdm(enumerate(val_dataloader), desc="Validation Batches", leave=False):
                frames = torch.squeeze(torch.tensor(
                    np.stack(frames[0])), dim=1).to(device)
                
                val_student_scores = student(frames)

                gt_score = gt_score[0].to(device)

                val_loss = criterion(val_student_scores, gt_score)
                running_val_loss += val_loss.item()


            epoch_val_loss = running_val_loss / len(val_dataloader)
            epoch_val_losses.append(epoch_val_loss)
            print(
                f"Epoch {epoch + 1}/{epochs}, Validation Loss: {epoch_val_loss:.5f}")
            logging.info(
                f"Epoch {epoch + 1}/{epochs}, Validation Loss: {epoch_val_loss:.5f}")

    return epoch_train_losses, epoch_val_losses
