#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random

import pandas as pd
from PIL import Image, ImageDraw
from torch.utils import data
from torchvision.transforms import functional as tf


root_dir = '/home/vahidn/projects/def-banire/vahidn/CLT/Data'

#images = [os.path.join(root_dir, 'images', x) for x in cows]
#images = [sorted([os.path.join(x, y) for y in os.listdir(x)]) for x in images]
#images = [x for cow in images for x in cow]

#labels = [os.path.join(root_dir, 'labels', x + '.csv') for x in cows]
#labels = [pd.read_csv(x, names=['x1', 'y1', 'x2', 'y2', 'x3', 'y3']).values.astype('float32') for x in labels]
#labels = [x / 4. for cow in labels for x in cow]


# In[2]:


class CLT(data.Dataset):

    def __init__(self, root_dir, cows, decode=False, scale=1, vflip=False, hflip=False, transform=None):
        self.decode = decode
        self.scale = scale
        self.vflip = vflip
        self.hflip = hflip
        self.transform = transform

        # self.grid_w = [x * 5 for x in range(1, 64)]
        # self.grid_h = [x * 5 for x in range(1, 36)]

        images = [os.path.join(root_dir, 'images', x) for x in cows]
        images = [sorted([os.path.join(x, y) for y in os.listdir(x)]) for x in images]
        images = [x for cow in images for x in cow]

        labels = [os.path.join(root_dir, 'labels', x + '.csv') for x in cows]
        labels = [pd.read_csv(x, names=['x1', 'y1', 'x2', 'y2', 'x3', 'y3']).values.astype('float32') for x in labels]
        labels = [x / 4. for cow in labels for x in cow]

        self.dataset = list(zip(images, labels))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        image = Image.open(image)

        # draw = ImageDraw.Draw(image)
        # for x in self.grid_w:
        #     draw.line([x, 0, x, 179], fill=128, width=0)
        #
        # for y in self.grid_h:
        #     draw.line([0, y, 319, y], fill=128, width=0)

        # label = label / 4.
        # label[1] += 70
        # label[3] += 70
        # label[5] += 70

        if self.decode:
            imseg = draw.get_triangle(label, scale=self.scale)
            if self.transform:
                image = self.transform(image)
                imseg = self.transform(imseg)
            return image, label, imseg.long()
        else:
            if self.transform:
                image = self.transform(image)
            return image, label


# In[3]:


images_dir = os.path.join(root_dir, 'images')
cows = [os.path.join(images_dir, x) for x in os.listdir(images_dir)]
cows = sorted([os.path.basename(x) for x in cows if os.path.isdir(x)])


# In[4]:


from torchvision.transforms import transforms
train_cows = cows[1:]
valid_cows = cows[:1]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4066, 0.4085, 0.4028], std=[0.1809, 0.1871, 0.1975])
])

train_dataset = CLT(root_dir=root_dir, cows=train_cows, decode=False, scale=1, transform=transform)
valid_dataset = CLT(root_dir=root_dir, cows=valid_cows, decode=False, scale=1, transform=transform)


# In[19]:


#images = [os.path.join(root_dir, 'images', x) for x in cows]
#images = [sorted([os.path.join(x, y) for y in os.listdir(x)]) for x in images]
#images = [x for cow in images for x in cow]

#labels = [os.path.join(root_dir, 'labels', x + '.csv') for x in cows]
#labels = [pd.read_csv(x, names=['x1', 'y1', 'x2', 'y2', 'x3', 'y3']).values.astype('float32') for x in labels]
#labels = [x / 4.  for cow in labels for x in cow]

#dataset = list(zip(images, labels))


# In[20]:


batch_size = 32

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


# In[21]:


#from matplotlib import image
#from matplotlib import pyplot as plt
  
# to read the image stored in the working directory
#data = image.imread(images[1000])
#x1, y1, x2, y2, x3, y3 = labels[1000]
# to draw a point on co-ordinate (200,300)
#plt.plot(x1, y1, marker='v', color="white")
#plt.plot(x2, y2, marker='v', color="white")
#plt.plot(x3, y3, marker='v', color="white")

#plt.imshow(data)
#plt.show()


# In[21]:


import torch
import torch.nn as nn
from torchvision import models

num_classes = 6
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

out_criterion = nn.SmoothL1Loss()
seg_criterion = nn.CrossEntropyLoss()
#out_criterion = nn.MSELoss()
# model = segnet.SegNet(channels=cfg['channels'], decode=cfg['decode']).to(device)
model = models.resnet18(pretrained=False, num_classes=num_classes).to(device)

# In[15]:


learning_rate = 1e-3
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# In[35]:


def iterate(ep, mode):
    
    decode = False
    if mode == 'train':
        model.train()
        loader = train_loader
    else:
        model.eval()
        loader = valid_loader

    num_samples = 0
    run_loss = 0.
    run_err = torch.zeros(3)

    monitor = tqdm(loader, desc=mode)
    for it in monitor:
        if decode:
            img, lbl, tri = it
            out, seg = model(img.to(device))
            out_loss = out_criterion(out, lbl.to(device))
            seg_loss = seg_criterion(seg, tri.squeeze(1).to(device))
            loss = out_loss + cfg['aux_ratio'] * seg_loss
        else:
            img, lbl = it
            out = model(img.to(device))
            loss = out_criterion(out, lbl.to(device))

        num_samples += lbl.size(0)
        run_loss += loss.item() * lbl.size(0)
        run_err += ((out.detach().cpu() - lbl) ** 2).view(-1, 3, 2).sum(2).sqrt().sum(0)

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #monitor.set_postfix(epoch=ep, loss=run_loss / num_samples, err=(run_err / num_samples).round().tolist(), avg=run_err.mean().item() / num_samples)
       # print('Train Loss: ', loss.item())
    if mode == 'train':
        scheduler.step()

    return run_loss / num_samples, run_err / num_samples


# In[36]:

#torch.save(model, '/home/vahidn/def-banire/vahidn/CLT/DensNetCLT.pt')
from tqdm import tqdm

num_epochs = 100

#if __name__ == '__main__':
best_avg = 1e16
best_ep = -1

train_loss = []
train_acc = []

valid_loss = []
valid_acc = []
patience = 0 
prev_l = 1e10

for epoch in range(num_epochs):
    l, err = iterate(epoch, 'train')
    train_loss.append(l)
    train_acc.append(err.mean())
    print('Epoch error in: ', epoch+1, 'is: ', err)
        #tqdm.write(f'Train | Epoch {epoch} | Error {err.tolist()}')
    with torch.no_grad():
        l, err = iterate(epoch, 'valid')
        valid_loss.append(l)
        valid_acc.append(err.mean())
        if err.mean() <= best_avg:
                #tqdm.write(f'NEW BEST VALIDATION | New Average {err.mean()} | Improvement {best_avg - err.mean()}')
            best_avg = err.mean()
            best_ep = epoch
    print('Epoch valid error in: ', epoch+1, 'is: ', err)

    if l > prev_l:
       patience += 1

    prev_l = l

    if patience > 8:
       torch.save(model, '/home/vahidn/projects/def-banire/vahidn/CLT/Res1.pt')
       break
                #torch.save(model.state_dict(), os.path.join(log_dir, '_' + str(err.mean().item()) + '_' + str(err.tolist()) + '_' + '.pt'))
            #tqdm.write(f'Valid | Epoch {epoch} | Error {err.tolist()} | Best Average {best_avg} | Best Epoch {best_ep}')
    
torch.save(model, '/home/vahidn/projects/def-banire/vahidn/CLT/Res1.pt')
# In[25]:


#l, err = iterate(0, 'train')


# In[26]:


#images, labels = next(iter(train_loader))


# In[31]:


#images[0].size()


# In[32]:


#print(len(dataset))


# In[ ]:




