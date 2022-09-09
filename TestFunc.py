

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os

import nibabel as nib
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import ndimage
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
# device='cuda:1' if torch.cuda.is_available() else 'cpu'
def getCenter(image, segmentation, i, j, k):

    sample=image[i-16:i+16+1,j-16:j+16+1,k-1:k+1+1]
    center=segmentation[i:i+1+1,j:j+1+1,k]
    
    return sample, center
    
def readAll(imgPath, betPath):
    
    positions=[]
    
    image = nib.load(imgPath).get_fdata()
    
    brainMask = nib.load(betPath).get_fdata()
    
    x,y,z=image.shape
    
    
    for z in range(image.shape[2]):
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):

#                 if annotation[x,y,z]==3 or  annotation[x,y,z]==6:
#                     annotation[x,y,z]=1

#                 if annotation[x,y,z]==5:
#                     annotation[x,y,z]=2

#                 if annotation[x,y,z]==4:
#                     annotation[x,y,z]=3


                if image[x,y,z]>200: image[x,y,z]=200
                if image[x,y,z]<-100: image[x,y,z]=-100
    
    image+=100
    image=image/300
    
    for k in range(1, z-1, 1):
        for i in range(17, x-17, 2):
            for j in range(17, y-17, 2):
                
                
                sample, center =getCenter(image, brainMask, i, j, k)
                if center.any():
                    positions.append((i,j,k))
#     return image, annotation
    return image, brainMask, positions, image.shape
    

def getPatch(image_full, brainMask, i, j, k):
    
    image, center=getCenter(image_full, brainMask, i, j, k)    
        
    
    return image, torch.tensor([i,j,k])


class NPHDataset(Dataset):
    def __init__(self, dataPath, betPath, name, Train=False):
        
        self.name=name
        self.image, self.brainMask, self.imgList, self.imageShape=readAll(dataPath, betPath)
        self.transform=transforms.ToTensor()

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        
#         return 0
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        i,j,k=self.imgList[idx]
        data, pos=getPatch(self.image, self.brainMask, i, j, k)
    
        image = self.transform(data)
        sample = {'img': image,
                  'pos': pos
                 }
        return sample

class MyModel(nn.Module):
    def __init__(self,ResNet, num_classes=5, num_outputs=4):
        super(MyModel, self).__init__()
 
        self.layer0=nn.Sequential(
            nn.Conv2d(3,64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False),
            
        )
        
        self.layer1=ResNet.layer1
        self.layer2=ResNet.layer2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.fc=nn.Linear(in_features=128, out_features=num_classes*num_outputs, bias=True)
        
    def forward(self, x):

        x=self.layer0(x)
        x=self.layer1(x)        
        x=self.layer2(x) 
        x=self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)              
        return x


# In[6]:

def test(imgName, model, test_loader, shape, device):

    model.eval()

    # Don't update model
#     print(len(test_loader))
    with torch.no_grad():
        predUnique={}
        targetUnique={}
        # Predict
        
        reconstructed=np.zeros(shape)
        probScore=np.zeros((2, shape[0], shape[1],shape[2]))
        # #####################
        #
        # 0 : vent - softmax
        # 1 : sub  - softmax
        # 2 : vent - sigmoid
        # 3 : sub  - sigmoid   
        # 
        # ######################    
        for batch_index, batch_samples in enumerate(test_loader):
            data, pos = batch_samples['img'].to(device, dtype=torch.float), batch_samples['pos']
            output = model(data)
            #Regression
            softmax=nn.Softmax(dim=1)

            output=torch.reshape(output,(output.shape[0], 5, 2, 2))
            
            output=softmax(output)


            pred=output.argmax(dim=1, keepdim=True)


            N=output.shape[0]

            for k in range(N):

                x, y, z=pos[k][0].item(), pos[k][1].item(), pos[k][2].item()
#### create heatmap ####
                probScore[0,x:x+1+1,y:y+1+1,z] = output[k,1,:,:].cpu()
                probScore[1,x:x+1+1,y:y+1+1,z] = output[k,3,:,:].cpu()

                reconstructed[x:x+1+1,y:y+1+1,z]=pred[k,0,:,:].cpu()

####save heatmap ####
        
        for slice in range(12,22):
            #softmax
            plt.figure(figsize = (10,10))
            ax = sns.heatmap(probScore[0,:,:,slice].reshape((shape[0],shape[1])))
            plt.savefig(os.path.join("heatmap",imgName+"_{}_vent".format(slice)))

            plt.figure(figsize = (10,10))
            ax = sns.heatmap(probScore[1,:,:,slice].reshape((shape[0],shape[1])))
            plt.savefig(os.path.join("heatmap",imgName+"_{}_sub".format(slice)))



    return reconstructed

def loadModel(modelPath, device):
    ResNet=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

    model = MyModel(ResNet, num_classes=5, num_outputs=4).to(device)
    model.load_state_dict(torch.load(modelPath,map_location=device))
    
    return model
    
def checkDevice(device):
    device=device if torch.cuda.is_available() else 'cpu'
    return device

def runTest(imgName, outputPath, dataPath, betPath, device, BS, model):
      
#     BS=200

    dataPath=os.path.join(dataPath,'{}.nii.gz'.format(imgName))     
 
    betPath=os.path.join(betPath,'{}_Mask.nii.gz'.format(imgName))
    
    testDataset=NPHDataset(dataPath, betPath, imgName,Train=False)
    test_loader = DataLoader(testDataset, batch_size=BS, num_workers=16, drop_last=False, shuffle=False)
    shape=testDataset.imageShape

    print('Start Running:', imgName)
    import time
    start = time.time()

    reconstructed=test(imgName, model, test_loader, shape, device)
    print(imgName, end=' ')
    result_noNoise=eliminateNoise(reconstructed, minArea=64)   
    filldots = cutoff(result_noNoise)        
    saveImage(filldots, os.path.join(outputPath, 'reconstructed_{}.nii.gz'.format(imgName)))    
#     # In[16]:

    end = time.time()
    print('Elapsed time:', end - start)

    ########result no noise 에서 ventricle 덩어리가 가장자리와 가까우면 subarachnoid로 바꿔주자
    #이제 location을 고려하는 수 밖에 업승ㅁ

    return 'reconstructed_{}.nii.gz'.format(imgName)
# In[ ]:
def saveImage(image, name):
    img = nib.Nifti1Image(image, np.eye(4))
    nib.save(img, name )

def cutoff(label):

    neighbors=[(1,1,0),(0,1,0),(-1,1,0),(-1,0,0),(-1,-1,0),(0,-1,0),(1,-1,0),(1,0,0)]
    surpos = [3,3,3,3,3,3,3,3]
    i, j, k=label.shape

    for z in range(k):
        for x in range(i):
            for y in range(j):
                if label[x,y,z] ==2: 
                    nei = []
                    for neighbor in neighbors:
                        if x-neighbor[0]<0 or x-neighbor[0]>=i: continue
                        if y-neighbor[1]<0 or y-neighbor[1]>=j: continue
                        nei.append(label[x-neighbor[0], y-neighbor[1],z-neighbor[2]])
                    if nei == surpos :
                        label[x,y,z] = 3
    return np.array(label)

def eliminateNoise(label, minArea=16):
    neighbors=[(-1,0),(1,0),(0,-1),(0,1)]
                
    seen=set()
    import heapq
    position=[]
    heapq.heapify(position)

    island=0
    newLabel=np.zeros(label.shape)
    i, j, k=label.shape
    for z in range(k):
        for x in range(i):
            for y in range(j):
                
                if (label[x,y,z]!=0) and (x,y,z) not in seen:
                    island+=1
                    area=0
                    curIsland=set()
                    seen.add((x,y,z))
                    curIsland.add((x,y,z))
                    heapq.heappush(position, (x,y,z))


                    while position:
                        cur=heapq.heappop(position)
                        area+=1


                        for neighbor in neighbors:

                            if cur[0]-neighbor[0]<0 or cur[0]-neighbor[0]>=i: continue
                            if cur[1]-neighbor[1]<0 or cur[1]-neighbor[1]>=j: continue
#                             if cur[2]-neighbor[2]<0 or cur[2]-neighbor[2]>=k: continue    

                            if label[cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]]==label[x,y,z] and (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]) not in seen:
                                seen.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]))
                                curIsland.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]))
                                heapq.heappush(position, (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]))



                    for (posX, posY, posZ) in curIsland: 
                        if area<minArea:
                            newLabel[posX, posY, posZ]=2
                        else:
                            newLabel[posX, posY, posZ]=label[x,y,z]


    return newLabel

def diceScore7slice(imgName, initial, gtPath,maxs):
    gtPath=os.path.join(gtPath,'Final_{}.nii.gz'.format(imgName))
    final = nib.load(gtPath).get_fdata()

    correct=0
    total=0
    TP=[0]*7
    FP=[0]*7
    FN=[0]*7
    
    for i in range(initial.shape[0]):
        for j in range(initial.shape[1]):
            for k in range(maxs-3,maxs+4,1):
                if final[i,j,k]==0 and initial[i,j,k]==0: continue
                total+=1
                if initial[i,j,k]==final[i,j,k]:
                    TP[int(final[i,j,k])]+=1

                    correct+=1

                else:
                    FN[int(final[i,j,k])]+=1
                    FP[int(initial[i,j,k])]+=1

    return correct, total, TP, FP, FN

def changeClass(annotation):
   
    for z in range(annotation.shape[2]):
        for x in range(annotation.shape[0]):
            for y in range(annotation.shape[1]):

                if annotation[x,y,z]==3:
                    annotation[x,y,z]=4

def fillHoles(imgName):
    image=nib.load('{}_Mask.nii.gz'.format(imgName)).get_fdata()
    for z in range(image.shape[2]):
        image[:,:,z]=ndimage.binary_fill_holes(image[:,:,z]).astype(int)

    saveImage(image, '{}_Mask.nii.gz'.format(imgName))  

if __name__=='__main__':
                    


    imgName='Norm_old_003_96yo'
    # from torchsummary import summary

    # summary(model, (3, 33, 33))
    runTest(imgName)

# %%
