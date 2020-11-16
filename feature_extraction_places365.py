# reference : https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
from torch.optim import lr_scheduler
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import os
from torch.utils.data import DataLoader  # Gives easier dataset management and creates mini batches

from tensorboardX import SummaryWriter
import time
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 3
num_classes = 3
learning_rate = 1e-3
batch_size = 128
num_epochs = 200

dir_made = 0 # flag for checkpoint dir
checkpoint_path = ""
def save_model(model_name, epoch, model):

    global dir_made
    global checkpoint_path
    if dir_made != 1:
        dir_name = time.strftime('%Y%m%d-%H%M%S-', time.localtime(time.time())) + model_name
        checkpoint_path = os.path.join('checkpoints', dir_name)
        if not os.path.exists(checkpoint_path):
            print('creating dir {}'.format(checkpoint_path))
            os.mkdir(checkpoint_path)
            dir_made = 1

    checkpoint_file_path = os.path.join(checkpoint_path, 'epoch-{}.pkl'.format(epoch))
    print('==> Saving checkpoint ... epoch {}'.format(epoch))
    torch.save(model, checkpoint_file_path)

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
        writer.add_scalar(mode + 'accuracy', float(num_correct.item() / num_samples) * 100, epoch)
    model.train()
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Load Data
transform = transforms.Compose([transforms.Resize((299, 299)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])

train_set = torchvision.datasets.ImageFolder(root="dataset/MITPlaces_indoor_3/train", transform=transform)
test_set = torchvision.datasets.ImageFolder(root="dataset/MITPlaces_indoor_3/test", transform=transform)
# print(dataset.__getitem__(18))
# train_set, test_set = torch.utils.data.random_split(dataset, [900, 100])
# train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=1000, shuffle=False)
# train_set = Subset(dataset, train_idx)
# test_set = Subset(dataset, test_idx)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# Model

# Model

# ### ResNext101, ResNet50, wide_resnet101_2
# model_conv = torchvision.models.resnet50(pretrained=True) # resnet101-fc.in_features
# print(model_conv)
# for param in model_conv.parameters():
#      param.requires_grad = False
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, num_classes)
# model_conv = model_conv.to(device)

# torchvision.models.shufflenet_v2_x0_5(pretrained=True)
# torchvision.models.mobilenet_v2(pretrained=True)

# ### MNasNet1_0, 0_5, 0_75(x), 1_3(x)
# model_conv = torchvision.models.mnasnet1_0(pretrained=True)
# for param in model_conv.parameters():
#      param.requires_grad = False
# num_ftrs = model_conv.classifier[1].in_features
# model_conv.classifier[1] = nn.Linear(num_ftrs, num_classes)
# model_conv = model_conv.to(device)

# ### GoogLeNet
# model_conv = torchvision.models.googlenet(pretrained=True)
# for param in model_conv.parameters():
#      param.requires_grad = False
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, num_classes)
# model_conv = model_conv.to(device)

# ### VGG19
# model_conv = torchvision.models.vgg19_bn(pretrained=True)
# for param in model_conv.parameters():
#      param.requires_grad = False
# num_ftrs = model_conv.classifier[6].in_features
# model_conv.classifier[6] = nn.Linear(num_ftrs, num_classes)
# model_conv = model_conv.to(device)

# ### DenseNet
# model_conv = torchvision.models.densenet121(pretrained=True)
# for param in model_conv.parameters():
#      param.requires_grad = False
# num_ftrs = model_conv.classifier.in_features
# model_conv.classifier = nn.Linear(num_ftrs, num_classes)
# model_conv = model_conv.to(device)

### Inception-v3 ### input size: 299x299
model_name = "Inception-v3"
model_conv = torchvision.models.inception_v3(pretrained=True)
for param in model_conv.parameters():
     param.requires_grad = False
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, num_classes)
model_conv = model_conv.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_conv.fc.parameters(), lr=learning_rate)
# optimizer = optim.Adam(model_conv.fc.parameters(), lr=learning_rate)
# 7 에폭마다 0.1씩 학습율 감소
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# tensorboard
writer = SummaryWriter(comment=model_name)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # print(data.shape
        # )
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward

        # Inception-v3
        outputs, aux_outputs = model_conv(data)
        loss1 = criterion(outputs, targets)
        loss2 = criterion(aux_outputs, targets)
        loss = loss1 + loss2 * 0.4
        # ### others
        # scores = model_conv(data)
        # loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}')
    if epoch % 100 == 99:
        writer.add_scalar('train_loss', sum(losses) / len(losses), epoch)
        # writer.add_scalar('test_loss', )
        check_accuracy(test_loader, model_conv, "Test", epoch)
        # save checkpoints
        save_model(model_name, epoch + 1, model_conv)





print("Checking accuracy on Training Set")
check_accuracy(train_loader, model_conv, "Train", 200)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model_conv, "Test", 200)

writer.close()