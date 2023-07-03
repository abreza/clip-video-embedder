import os
import json
from tqdm import tqdm
import glob
import logging
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.io import read_image
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

annotation_file = '/home/p_haghighi/clip-video-embedder/datasets/ActivityNet/activity_net.v1-3.min.json'
saved_model_path = '/media/external_10TB/10TB/p_haghighi/saved_models/finetuned_resnet50_activitynet/'

# create a subdirectory with a name based on current time\
def get_time():
    time_offset = 3600 * (4) + 60 * (-2) + 1 * (-22)
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time() + time_offset))
    return timestamp


subdir_path = os.path.join(saved_model_path, get_time())
os.makedirs(subdir_path, exist_ok=True)

# Set up logging
output_log_path = 'log_files/finetuned_resnet50_activitynet'
os.makedirs(output_log_path, exist_ok=True)

log_file = os.path.join(output_log_path, get_time()+'.txt')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO
)

# Define transforms for the training and validation sets
data_transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class ActivityNetDataset(Dataset):
    def __init__(self, img_dir, annotations_file, subset, only_segment_frames=False, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.only_segment_frames = only_segment_frames  # add this line
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)["database"]
        self.samples = []
        self.labels = []

        # Collect all frames and their labels
        for video_id, annotation in self.annotations.items():
            if annotation["subset"] == subset:
                video_dir = os.path.join(self.img_dir, 'v_'+video_id)
                video_frames = glob.glob(video_dir + '/*.jpg')  # get all frames of the video
                video_label = annotation["annotations"][0]["label"] if subset != 'testing' else ''

                for frame in video_frames:
                    frame_number = int(frame.split('_frame')[-1].split('.jpg')[0])  # get frame number from file name
                    if not self.only_segment_frames or any(a['segment'][0] <= frame_number < a['segment'][1] for a in annotation['annotations']):
                        self.samples.append(frame)
                        self.labels.append(video_label)

        # Encode labels to integers for train and val, not needed for test
        if subset != 'testing':
            self.le = LabelEncoder()
            self.le.fit(self.labels)
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = read_image(img_path).float()
        label = self.le.transform([label])[0] if label else -1  # use -1 or any suitable placeholder for test samples
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_latest_checkpoint(model):
    # Find the latest saved model checkpoint
    checkpoint_files = glob.glob(os.path.join(saved_model_path, 'model_epoch_*.pth'))

    if checkpoint_files:
        latest_checkpoint_file = max(checkpoint_files, key=os.path.getctime)

        file_name_without_extension = os.path.splitext(latest_checkpoint_file)[0]
        start_epoch = int(file_name_without_extension.split('_')[-1]) + 1

        model_state_dict = torch.load(latest_checkpoint_file)
        model.load_state_dict(model_state_dict)

    else:
        start_epoch = 0
        print('No saved checkpoints found, starting from scratch')

    return model, start_epoch


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=25):
    
    # model, start_epoch = load_latest_checkpoint(model)
    start_epoch = 0

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    model = model.to(device)

    
    for epoch in range(start_epoch, num_epochs):
        
        logging.info('Epoch {}/{}'.format(epoch+1, num_epochs))

        torch.cuda.empty_cache()
        # Training phase
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in tqdm(train_loader, 'Epoch {}/{} - Training'.format(epoch+1, num_epochs)):
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        logging.info(f'{get_time()}: Train Loss: {epoch_loss:.4f}\tAcc: {epoch_acc:.4f}')
        print(f'{get_time()}: Train Loss: {epoch_loss:.4f}\tAcc: {epoch_acc:.4f}')

        # Save model
        model_path = os.path.join(saved_model_path, 'model_epoch_{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_path)
        
        # Validation phase
        model.eval()   # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0


        for inputs, labels in tqdm(val_loader, 'Epoch {}/{} - validation'.format(epoch+1, num_epochs)):
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
         
        logging.info(f'{get_time()}: Val Loss: {epoch_loss:.4f}\tAcc: {epoch_acc:.4f}')
        print(f'{get_time()}: Val Loss: {epoch_loss:.4f}\tAcc: {epoch_acc:.4f}')



# Create training, validation and test datasets
train_dataset = ActivityNetDataset('/media/external_10TB/10TB/p_haghighi/ActivityNet/sampled_frames/train', annotation_file, 'training', data_transform)
val_dataset = ActivityNetDataset('/media/external_10TB/10TB/p_haghighi/ActivityNet/sampled_frames/val_1', annotation_file, 'validation', data_transform)
# test_dataset = ActivityNetDataset('./sampled_frames/test', annotation_file, 'testing', data_transform)

# Create training, validation and test dataloaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Load pretrained ResNet-50 model
model_ft = models.resnet50(weights='IMAGENET1K_V1')
model_ft.fc = nn.Linear(model_ft.fc.in_features, 200) 

# Define the loss function, optimizer and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

print('Training started.')
# Train and evaluate
train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, train_loader, val_loader, num_epochs=100)