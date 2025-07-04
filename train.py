import sys
sys.setrecursionlimit(15000)
import os
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam,SGD
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
from utils import AverageMeter, Logger, make_optimizer, DeepSupervision
import yaml
import os
from models import PSTA
import os
from tddfa_util import (
    load_model, _parse_param, similar_transform,
    ToTensorGjz, NormalizeGjz,_load
)
from torchvision.transforms import Compose
import numpy as np
from models.mobilenet_v3 import MobileNet
import random
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from models.res2net_v1b import res2net101_v1b,Res2Net


parser = argparse.ArgumentParser()
parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight_decay')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
parser.add_argument('--gpu_id', type=str, default='cuda:0', help='GPU ID')
parser.add_argument('--resume', type=int, default=0, help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--outf', default='/media/sd01/2e14da1a-3bd5-489a-8fb0-e2522dd64c02/xtt/test', help='folder to output model checkpoints')

opt = parser.parse_args()
print(opt)

class MyDatasetWDF(Dataset):
    def __init__(self, root_dir,stride=8, transforms1_=None,transforms2_=None,clip_len=4,mode='train'):
        self.root_dir = root_dir
        self.transforms1_ = transforms1_
        self.toPIL = transforms.ToPILImage()
        self.data = []
        self.clip_len=clip_len
        self.stride=stride
        self.transforms2_=transforms2_
        self.td_size=120
        

     
        for base, subdirs, files in os.walk(self.root_dir):
         
            if len(files) < self.stride * self.clip_len:  #
                continue
            data = {}
            video = []
            files.sort()
            for i, f in enumerate(files):
                if f.endswith('.png'):
                    data_dict = {}
                    data_dict['frame'] = os.path.join(base, f)
                    data_dict['index'] = i
                    video.append(data_dict)
            if video==[]:
                continue
            data['video'] = video

            data['label'] = 1 if 'real' in base else 0
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        video = self.data[idx]['video']
        label = self.data[idx]['label']
     

        clip_start =1
        clip = video[clip_start: clip_start + (self.clip_len * self.stride): self.stride]


        if self.transforms1_:
            trans1_clip = []
     
            for frame in clip:
       
                # print(frame['frame'])
                frame = Image.open(frame['frame'])
                frame = self.transforms1_(frame)  # tensor [C x H x W]

                trans1_clip.append(frame)
            td_clip= torch.stack(trans1_clip)
        



        if self.transforms2_:
            trans2_clip = []
            #seed = random.random()
            for frame in clip:
         
                frame = Image.open(frame['frame'])
                frame = self.transforms2_(frame)  # tensor [C x H x W]

                trans2_clip.append(frame)
         
            text_clip= torch.stack(trans2_clip)

        return td_clip, torch.tensor(int(label)),text_clip


class MyDatasetFF(Dataset):
    def __init__(self, root_dir,stride=8, transforms1_=None,transforms2_=None,clip_len=4,mode='train'):
        self.root_dir = root_dir
        self.transforms1_ = transforms1_
        self.toPIL = transforms.ToPILImage()
        self.data = []
        self.clip_len=clip_len
        self.stride=stride
        self.transforms2_=transforms2_
        self.td_size=120
        

           
        for base, subdirs, files in os.walk(self.root_dir):
         
            if len(files) < self.stride * self.clip_len:  #
                continue
            data = {}
            video = []
            files.sort()
            for i, f in enumerate(files):
                if f.endswith('.jpg'):
                    data_dict = {}
                    data_dict['frame'] = os.path.join(base, f)
                    data_dict['index'] = i
                    video.append(data_dict)
            if video==[]:
                continue
            data['video'] = video

            data['label'] = 1 if 'real' in base else 0
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        video = self.data[idx]['video']
        label = self.data[idx]['label']
      
        clip_start =1
        clip = video[clip_start: clip_start + (self.clip_len * self.stride): self.stride]

        if self.transforms1_:
            trans1_clip = []
 
            for frame in clip:
       
                # print(frame['frame'])
                frame = Image.open(frame['frame'])
                frame = self.transforms1_(frame)  # tensor [C x H x W]

                trans1_clip.append(frame)
         
     
            td_clip= torch.stack(trans1_clip)
        




        if self.transforms2_:
            trans2_clip = []
  
            for frame in clip:
         
                frame = Image.open(frame['frame'])
                frame = self.transforms2_(frame)  # tensor [C x H x W]

                trans2_clip.append(frame)
          
            text_clip= torch.stack(trans2_clip)

        return td_clip, torch.tensor(int(label)),text_clip




class MyDatasetCDF(Dataset):
    def __init__(self, root_dir,stride=8, transforms1_=None,transforms2_=None,clip_len=4,mode='train'):
        self.root_dir = root_dir
        self.transforms1_ = transforms1_
        self.toPIL = transforms.ToPILImage()
        self.clip_len=clip_len
        self.stride=stride
        self.transforms2_=transforms2_
        self.td_size=120
        self.data=[]

       
        for base, subdirs, files in os.walk(self.root_dir):
   

            if len(files) < self.stride * self.clip_len:  #
                continue
            data = {}
            video = []
            files.sort()
            for i, f in enumerate(files):
                if f.endswith('.png'):
                    data_dict = {}
                    data_dict['frame'] = os.path.join(base, f)
                    data_dict['index'] = i
                    video.append(data_dict)
            data['video'] = video

            data['label'] = 1 if 'real' in base else 0
            self.data.append(data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        video = self.data[idx]['video']
        label = self.data[idx]['label']


        clip_start =0
        clip = video[clip_start: clip_start + (self.clip_len * self.stride): self.stride]
      
        if self.transforms1_:
            trans1_clip = []
            #seed = random.random()
            for frame in clip:
             
                frame = Image.open(frame['frame'])
                frame = self.transforms1_(frame)  # tensor [C x H x W]

                trans1_clip.append(frame)
          
            td_clip= torch.stack(trans1_clip)
        


        if self.transforms2_:
            trans2_clip = []
          
            for frame in clip:
                
                frame = Image.open(frame['frame'])
                frame = self.transforms2_(frame)  # tensor [C x H x W]

                trans2_clip.append(frame)
        
            text_clip= torch.stack(trans2_clip)

        return td_clip,  torch.tensor(int(label)),text_clip







transform_normalize = NormalizeGjz(mean=127.5, std=128)
train_transforms1= transforms.Compose([
transforms.Resize((120,120)),
transforms.RandomHorizontalFlip(p=0.3),
transforms.ToTensor(),
transform_normalize
])
test_transforms1= Compose([
    transforms.Resize((120,120)),
    transforms.ToTensor(), transform_normalize])

test_transforms2= transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
   
    transforms.Normalize(mean=(0.643, 0.466, 0.387), std=(0.259, 0.220, 0.203))
])

train_transforms2 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.643, 0.466, 0.387), std=(0.259, 0.220, 0.203))
])




trainset = MyDatasetWDF('/media/sd01/2e14da1a-3bd5-489a-8fb0-e2522dd64c02/xtt/dataset/WildDeepfake/train',
                    transforms1_=train_transforms1,transforms2_=train_transforms2)  # 10-frame number
train_bs=8
        
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True,drop_last=True,pin_memory=False)


testset=MyDatasetWDF('/media/sd01/2e14da1a-3bd5-489a-8fb0-e2522dd64c02/xtt/dataset/WildDeepfake/test',transforms1_=test_transforms1,transforms2_=test_transforms2)
test_bs=8
testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=True,drop_last=True,pin_memory=False)







if __name__ == "__main__":
    torch.cuda.empty_cache()
    device=torch.device(opt.gpu_id)
    fusionmodel=PSTA(num_classes=2, seq_len=8).to(device)

    tddfa= MobileNet(widen_factor=1,num_classes=62).to(device)
    tddfa = load_model(tddfa,'configs/mb1_120x120.pth')
    tddfa.train(mode=True)

    modelcnn=res2net101_v1b().to(device)
    model_path='/media/sd01/2e14da1a-3bd5-489a-8fb0-e2522dd64c02/xtt/modelwild.pth'
    modelcnn.load_state_dict(torch.load(model_path))



    xent = torch.nn.CrossEntropyLoss().to(device)
    tent = torch.nn.CrossEntropyLoss().to(device)

    lr=opt.lr
    optimizer = Adam([
        {'params': tddfa.parameters(), 'lr': lr},
        {'params': modelcnn.parameters(), 'lr': lr},
        {'params': fusionmodel.parameters(), 'lr': lr}
        ], lr=lr, betas=(opt.beta1,opt.beta2),weight_decay=opt.weight_decay)  




    resume=opt.resume
    outf=opt.outf
    if  resume > 0:
        text_writer = open(os.path.join(outf, 'train.csv'), 'a')
    else:
        text_writer = open(os.path.join(outf, 'train.csv'), 'w')
    if resume > 0:
        tddfa.load_state_dict(torch.load(os.path.join(outf, 'tddfa_' + str(resume) + '.pt')),strict=False)
        tddfa.train(mode=True)

        modelcnn.load_state_dict(torch.load(os.path.join(outf, 'modelcnn_' + str(resume) + '.pt')))
        modelcnn.train(mode=True)
        fusionmodel.load_state_dict(torch.load(os.path.join(outf,'capsule_' + str(resume) + '.pt')))
        fusionmodel.train(mode=True)
        optimizer.load_state_dict(torch.load(os.path.join(outf,'optim_' + str(resume) + '.pt')))


        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(device)






    epochs=opt.niter
    xent = torch.nn.CrossEntropyLoss().to(device)
    tent = torch.nn.CrossEntropyLoss().to(device)
    for epoch in range(resume+1, epochs+1):
        torch.cuda.empty_cache()

        print(epoch)
        count = 0
        loss_train = 0
        loss_test = 0
    


        tol_label = np.array([], dtype=float)
        tol_pred = np.array([], dtype=float)

        for batch_idx, data0 in enumerate(trainloader, 1):
            td, labels_data,text = data0

            td=td.to(device)
            text=text.to(device)
            labels_data=labels_data.to(device)
            # print(td.shape)
            # print(text.shape)
            #(b,t,c,h,w)
            optimizer.zero_grad()
            b,t,c,h1,w1=td.shape
            b,t,c,h2,w2=text.shape
            origin_td=td
            td = td.contiguous().view(b * t, c, w1, h1)
            text=text.contiguous().view(b * t, c, w2, h2)

        
            y = torch.repeat_interleave(labels_data , 4)
    
            tdfeature=tddfa(td)
            cnnout,prediction1=modelcnn(text)
            # print(tdfeature.shape)
            # print(cnnout.shape)

            
            # print(predicted_labels.shape,predicted_labels)

            outputs, features = fusionmodel(origin_td,tdfeature,cnnout)
            if isinstance(outputs, (tuple, list)):
                xent_loss = DeepSupervision(xent, outputs, labels_data )

            else:
                xent_loss = xent(outputs, labels_data ).to(device)



            loss2=xent(prediction1,y)
            loss_dis = 0.7*xent_loss+0.3*loss2
            loss_dis_data = loss_dis.item()
            loss_dis.backward()
            # print(loss_dis)
            # print(tacc)
            optimizer.step() 

            loss_train += loss_dis_data
            count += 1
            


        loss_train/=count


        print(loss_train)



        #do checkpointing & validation
        torch.save(tddfa.state_dict(), os.path.join(outf, 'tddfa_%d.pt' % epoch))
        torch.save(modelcnn.state_dict(), os.path.join(outf, 'modelcnn_%d.pt' % epoch))
        torch.save(fusionmodel.state_dict(), os.path.join(outf, 'capsule_%d.pt' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(outf, 'optim_%d.pt' % epoch))
        print("saved",epoch,"")
        tddfa.eval()
        modelcnn.eval()
        fusionmodel.eval()

        tol_label = np.array([], dtype=float)
        tol_pred = np.array([], dtype=float)

        count = 0
        with torch.no_grad():
            for batch_idx, test_data0 in enumerate(testloader, 1):
                td, labels_data,text = test_data0


                td=td.to(device)
                text=text.to(device)
                labels_data=labels_data.to(device)
                # print(td.shape)
                # print(text.shape)
                #(b,t,c,h,w)
                optimizer.zero_grad()
                b,t,c,h1,w1=td.shape
                b,t,c,h2,w2=text.shape
                origin_td=td
                td = td.contiguous().view(b * t, c, w1, h1)
                text=text.contiguous().view(b * t, c, w2, h2)

        
                y = torch.repeat_interleave(labels_data , 4)
            
                tdfeature=tddfa(td)
                cnnout,teprediction1=modelcnn(text)
        
                outputs, features = fusionmodel(origin_td,tdfeature,cnnout)
                # print(len(outputs),outputs[0].shape)#torch.Size([16, 2]


                if isinstance(outputs, (tuple, list)):
                    xent_loss = DeepSupervision(xent, outputs, labels_data).to(device)

                else:
                    xent_loss = xent(outputs, labels_data).to(device)


                loss2=xent(teprediction1,y)
            
                loss_dis =0.7*xent_loss+0.3*loss2
                loss_dis_data = loss_dis.item()

    
                loss_test+=loss_dis_data
                count += 1
       


            loss_test /= count

        
        
        print('[Epoch %d] Train loss: %.4f   | Test loss: %.4f '
        % (epoch, loss_train, loss_test))

        text_writer.write('%d,%.4f,%.4f\n'
        % (epoch, loss_train, loss_test))

        text_writer.flush()

        tddfa.train()
        modelcnn.train()
        fusionmodel.train(mode=True)

    text_writer.close()


   



