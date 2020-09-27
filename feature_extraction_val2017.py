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

from sklearn.model_selection import train_test_split

from tensorboardX import SummaryWriter

class PlaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # img_path = os.path.join(self.img_dir, str(self.annotations.iloc[index, 0]) + '.jpg')
        img_path = os.path.join(self.img_dir, str(self.annotations.iloc[index, 0]))
        image = io.imread(img_path)
        image = Image.fromarray(image)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        # print(img_path)

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 3
num_classes = 3
learning_rate = 1e-3
batch_size = 32
num_epochs = 200

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Load Data and Augment
rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
'''
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(rgb_mean, rgb_std),
])'''
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_set = PlaceDataset(csv_file='data/csv/gt5000.csv', img_dir='./dataset/images/val2017',
                             transform=transform)
test_set = PlaceDataset(csv_file='data/csv/movie_gt3.csv', img_dir='data/test3',
                             transform=transform)
print(len(train_set), len(test_set))
# train_set, test_set = torch.utils.data.random_split(dataset, [400, 100])
# train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=100, shuffle=False)
# train_set = Subset(dataset, train_idx)
# test_set = Subset(dataset, test_idx)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# Model
model_conv = torchvision.models.resnet101() # resnet101-fc.in_features #pretrained=True
# print(model_conv)
# for param in model_conv.parameters():
#     param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, num_classes)

model_conv = model_conv.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_conv.parameters(), lr=learning_rate)
# optimizer = optim.Adam(model_conv.fc.parameters(), lr=learning_rate)
# 7 에폭마다 0.1씩 학습율 감소
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# tensorboard
writer = SummaryWriter(comment=model_conv.__class__.__name__)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model_conv(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}')
    if epoch % 5 == 0:
        writer.add_scalar('train_loss', sum(losses) / len(losses), epoch)

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
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
        print(num_correct/num_samples)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
        writer.add_scalar('accuracy', float(num_correct) / float(num_samples) * 100, num_correct / num_samples)
    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model_conv)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model_conv)

writer.close()