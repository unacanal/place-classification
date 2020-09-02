from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

#transform PIL images into tensors
import torchvision.transforms as transforms
import torchvision.models as models

#to deep copy the models; system package
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#PIL image 0~255 -> tensor value 0~1로 바꿔야함
#also, resize하고 같은dimensions이게 설정
#0~255값을 networks에 넣어주면 feature map는 unable sense됨

#desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128 # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize((imsize,imsize)), #scale imported image
    transforms.ToTensor() #transform it into a torch tensor
])

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0) # 차원 늘려줌
    return image.to(device, torch.float)

style_img = image_loader("./images/1-01.jpg")
content_img = image_loader("./images/1-02.jpg")

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

unloader = transforms.ToPILImage() # reconvert into PIL image
#display 하려고

plt.ion() #그림 갱신 ? 그림 업데이트

def imshow(tensor, title=None):
    image = tensor.cpu().clone() #we clone the tensor to not do changes on it
    image = image.squeeze(0) #remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) #pause a bit so that plots are updated

plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

class ContentLoss(nn.Module): #contentloss는 pytorch loss function이 아니라서 autograd function recompute해줘야함
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        #we 'detach' the target content from the tree used
        #to dynamically compute the gradient : this is a stated value,
        # not a variable, Otherwise the forward method of the criterion will throw an error
        self.target = target.detach() #target과 분리하여 새로운 target 할당. gradient 추적되지 않기위해

    def forward(self, input): #forward(input, target)?
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d =input.size() # a=batch sieze(=1)
    #b=number of feature maps
    #(c,d)=dimensions of a f. map (N=c*d)
    #이미 vgg를 통과한 feature겠지
    features = input.view(a*b, c*d) #resize F_XL into \hat F_XL
    G = torch.mm(features, features.t()) #compute the gram product 곱하고더함

    #we 'normalize' the values of the gram matrix
    #by dividing by the number of element in each feature maps.
    return G.div(a*b*c*d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

#VGG 19 사용. feature과 classifier중 feature만 가져와야함
#training과 evaluation이 다른경우가 있어서 우리는 무조건 .eval().

cnn = models.vgg19(pretrained=True).features.to(device).eval()

#VGG net은 각 채널이 mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]으로 normalize됨
#net에 send하기 전에 normalize
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.299, 0.224, 0.225]).to(device)

#create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        #.view the mean as std to make them [C X 1 X 1] so that they can
        # directly work with image Tensor of shape [B X C X H X W].
        # B is batch size. C is number of channels. H is height W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        #print(mean.shape)

    def forward(self, img):
        #normalize img
        #print(img.shape)
        return (img-self.mean)/self.std

#vgg.feature의 conv 가져와야함
#desired depth layers to compute style/content losses :
content_layers_default=['conv_4'] #4번째 layer
style_layers_default=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'] #style은 5개 다봄

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img,
                               content_layers=content_layers_default, style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    #normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device) #class 초기화

    #just in order to have an iterable access to or list of content/style losses
    content_losses = []
    style_losses = []

    #assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    #to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization) # 첫번째로 normalization

    i=0 #increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i+=1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            #The in-place version doesn't play very nicely with the ContentLoss
            #and StyleLoss we insert below. So we replace with out-of-place
            #ones here
            layer = nn.ReLU(inplace=False) #inplace=true : modify input directly, without output
        elif isinstance(layer, nn.MaxPool2d):
            name ='pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer:{}'.format(layer.__class__.__name__))

        model.add_module(name,layer)

        if name in content_layers:
            #add content loss:
            target = model(content_img).detach() #model에 content_img넣은 결과 -> feature map
            content_loss = ContentLoss(target) # feature map target 초기화
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            #add style loss:
            target_feature = model(style_img).detach() #model에 style_img 넣은 결과
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    #now we trim off the layers after the last content and style losses
    for i in range(len(model) -1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i+1)]

    return model, style_losses, content_losses

input_img = content_img.clone()
#if you want to use white noise instead uncomment the below line:
#input_img = torch.randn(content_img.data.size(), device =device)

#add the original input image to the figure:
plt.figure()
imshow(input_img,  title="Input Image")

#Gradient Descent
#우리는 input image 자체를 train시키고 싶음. -> L-BFGS algorithm
#BFGS 알고리즘의 목적은 f(x)를 제한 조건이 없는 실수 벡터 x에 대해서 최소화 시키는 것
def get_input_optimizer(input_img):
    #this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

#input training과정중 0~1을 벗어나면 고쳐주어야함 -> clamp함수(자름)
def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img,
                       num_steps=500, style_weight=1000000, content_weight=1):
    """Run the style trnasfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                                                     style_img, content_img)
    optimizer = get_input_optimizer(input_img)
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            #correct the values of updated input image
            input_img.data.clamp_(0,1)

            optimizer.zero_grad()
            model(input_img)
            style_score =0
            content_score =0

            for sl in style_losses:
                style_score +=sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss:{:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
                print()

            return style_score+content_score

        optimizer.step(closure)

    #a last correction...
    input_img.data.clamp_(0,1)

    return input_img

#Finally... we can run the algorithm
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

plt.ioff()
plt.show()