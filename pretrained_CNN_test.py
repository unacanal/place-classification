### only inferencing
### with 100 images of COCO test2017 dataset

# reference : https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html
# ResNet, GoogLeNet
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
from torch.optim import lr_scheduler
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import os
import pandas as pd
from skimage import io
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader  # Gives easier dataset managment and creates mini batches
import time

from sklearn.model_selection import train_test_split

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

class PlaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, str(self.annotations.iloc[index, 0]) + '.jpg')
        # img_path = os.path.join(self.img_dir, str(self.annotations.iloc[index, 0]))
        image = io.imread(img_path)
        image = Image.fromarray(image).convert('RGB')
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        # print(img_path)
        if self.transform:
            image = self.transform(image)

        return (image, y_label)

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model, mode, epoch):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        # print(num_correct/num_samples)
        print("-------"+mode+"-------")
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct.item() / num_samples) * 100:.2f}')
        print(num_correct.item())

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 3
num_classes = 3
learning_rate = 1e-3
batch_size = 128
num_epochs = 200

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Load Data and Augment
rgb_mean = (0.4914, 0.4822, 0.4465)
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_set = PlaceDataset(csv_file='data/csv/movie_gt3.csv', img_dir='data/test3', transform=transform)

print(len(test_set))
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=16, shuffle=False, pin_memory=True)

# Model
# ### ResNext101, ResNet50, wide_resnet101_2
# model_conv = torchvision.models.wide_resnet101_2(pretrained=True) # resnet101-fc.in_features
# print(model_conv)
# for param in model_conv.parameters():
#      param.requires_grad = False
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, num_classes)
# model_conv = model_conv.to(device)

# torchvision.models.shufflenet_v2_x0_5(pretrained=True)
# torchvision.models.mobilenet_v2(pretrained=True)
### MNasNet1_0, 0_5, 0_75(x), 1_3(x)
model_conv = torchvision.models.mnasnet0_5(pretrained=True)
for param in model_conv.parameters():
     param.requires_grad = False
num_ftrs = model_conv.classifier[1].in_features
model_conv.classifier[1] = nn.Linear(num_ftrs, num_classes)
model_conv = model_conv.to(device)

# ### GoogLeNet
# model_conv = torchvision.models.googlenet(pretrained=True)
# for param in model_conv.parameters():
#      param.requires_grad = False
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, num_classes)
# model_conv = model_conv.to(device)

# ### VGG19
# model_conv = torchvision.models.vgg19(pretrained=True)
# for param in model_conv.parameters():
#      param.requires_grad = False
# num_ftrs = model_conv.classifier[6].in_features
# model_conv.classifier[6] = nn.Linear(num_ftrs, num_classes)
# model_conv = model_conv.to(device)

# ### DenseNet
# model_conv = torchvision.models.densenet161(pretrained=True)
# for param in model_conv.parameters():
#      param.requires_grad = False
# num_ftrs = model_conv.classifier.in_features
# model_conv.classifier = nn.Linear(num_ftrs, num_classes)
# model_conv = model_conv.to(device)

# ### Inception-v3
# model_conv = torchvision.models.inception_v3(pretrained=True)
# for param in model_conv.parameters():
#      param.requires_grad = False
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, num_classes)
# model_conv = model_conv.to(device)

model = torch.load(PATH)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model, "Test", 200)
