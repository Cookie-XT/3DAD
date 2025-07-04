import sys
import csv
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
from utils import AverageMeter, Logger, make_optimizer, DeepSupervision,DeepSupervision_acc2,accuracy_score1
import yaml
import os
from models import PSTA


from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score
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
parser.add_argument('--gpu_id', type=str, default='cuda:0', help='GPU ID')
parser.add_argument('--outf', default='test', help='folder to save results')
parser.add_argument('--random', action='store_true', default=False, help='enable randomness for routing matrix')


opt = parser.parse_args()
print(opt)

class MyDatasetWDP(Dataset):
    def __init__(self, root_dir,stride=8, transforms1_=None,transforms2_=None,clip_len=4,mode='train'):
        self.root_dir = root_dir
        self.transforms1_ = transforms1_

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
        cname=clip[1]['frame'].rsplit('/',2)[0]
        names=cname.rsplit('/',3)
        cname=names[1]+'_'+names[2]+'_'+names[3]

        if self.transforms1_:
            trans1_clip = []
    
            for frame in clip:
               
                frame = Image.open(frame['frame'])
                frame = self.transforms1_(frame)  # tensor [C x H x W]
                trans1_clip.append(frame)
   
            td_clip= torch.stack(trans1_clip)
        


    
        if self.transforms2_:
            trans2_clip = []
  
            for frame in clip:
                frame = Image.open(frame['frame'])
                frame = self.transforms2_(frame)
                trans2_clip.append(frame)
     
            text_clip= torch.stack(trans2_clip)

        return td_clip, torch.tensor(int(label)),text_clip,cname


class MyDatasetCDF(Dataset):
    def __init__(self, root_dir,stride=8, transforms1_=None,transforms2_=None,clip_len=4,mode='train'):
        self.root_dir = root_dir
        self.transforms1_ = transforms1_

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
        cname=clip[1]['frame'].rsplit('/',4)
        cname=cname[1]+'_'+cname[2]
        # print(cname)
  
        if self.transforms1_:
            trans1_clip = []
        
            for frame in clip:
            
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

        return td_clip, torch.tensor(int(label)),text_clip,cname

class MyDatasetff(Dataset):
    def __init__(self, root_dir,stride=8, transforms1_=None,transforms2_=None,clip_len=4,mode='train'):
        self.root_dir = root_dir
        self.transforms1_ = transforms1_
      
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
        cname=clip[1]['frame'].rsplit('/',4)
        cname=cname[1]+'_'+cname[2]
    
        # print(cname)
        
        if self.transforms1_:
            trans1_clip = []
      
            for frame in clip:
             
                frame = Image.open(frame['frame'])
                frame = self.transforms1_(frame)  # tensor [C x H x W]
                trans1_clip.append(frame)
       
            td_clip= torch.stack(trans1_clip)
    
  
        if self.transforms2_:
            trans2_clip = []
           
            for frame in clip:
                #random.seed(seed)
                frame = Image.open(frame['frame'])
                frame = self.transforms2_(frame)  # tensor [C x H x W]

                trans2_clip.append(frame)
     
            text_clip= torch.stack(trans2_clip)

        return td_clip, torch.tensor(int(label)),text_clip,cname


transform_normalize = NormalizeGjz(mean=127.5, std=128)
test_transforms1= Compose([
    transforms.Resize((120,120)),
    transforms.ToTensor(), transform_normalize])

test_transforms2= transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.643, 0.466, 0.387), std=(0.259, 0.220, 0.203))
])


testset=MyDatasetWDP('/media/sd01/2e14da1a-3bd5-489a-8fb0-e2522dd64c02/xtt/dataset/WildDeepfake/test',transforms1_=test_transforms1,transforms2_=test_transforms2)
test_bs=8
testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=True,drop_last=True,pin_memory=False)






if __name__ == '__main__':

    device=torch.device(opt.gpu_id)


    tddfa= MobileNet(widen_factor=1,num_classes=62).to(device)
    tddfa_path='/media/sd01/2e14da1a-3bd5-489a-8fb0-e2522dd64c02/xtt/finalmodel/pre-train/wdf/tddfa.pt'
    tddfa.load_state_dict(torch.load(tddfa_path))
    tddfa.eval()

    modelcnn=res2net101_v1b().to(device)
    model_path='/media/sd01/2e14da1a-3bd5-489a-8fb0-e2522dd64c02/xtt/finalmodel/pre-train/wdf/modelcnn.pt'
    modelcnn.load_state_dict(torch.load(model_path))
    modelcnn.eval()

    fusionmodel=PSTA(num_classes=2, seq_len=8).to(device)
    fusionmodel_path='//media/sd01/2e14da1a-3bd5-489a-8fb0-e2522dd64c02/xtt/finalmodel/pre-train/wdf/fusion.pt'
    fusionmodel.load_state_dict(torch.load(fusionmodel_path))
    fusionmodel.eval()



    tol_label = np.array([], dtype=float)
    tol_pred = np.array([], dtype=float)

    loss_test=0.0

    count = 0
    with torch.no_grad():
        for batch_idx, test_data0 in enumerate(testloader, 1):
            td, labels_data,text,img_name = test_data0
            td=td.to(device)
            text=text.to(device)
            labels_data=labels_data.to(device)
            # print(td.shape)
            # print(text.shape)
            #(b,t,c,h,w)
            b,t,c,h1,w1=td.shape
            b,t,c,h2,w2=text.shape
            origin_td=td
            td = td.contiguous().view(b * t, c, w1, h1)
            text=text.contiguous().view(b * t, c, w2, h2)
            ##将labels重复4次
            y = torch.repeat_interleave(labels_data , 4)

            tdfeature=tddfa(td)
            cnnout,teprediction1=modelcnn(text)

            outputs, features = fusionmodel(origin_td,tdfeature,cnnout)
            # print(len(outputs),outputs[0].shape)#torch.Size([16, 2]

            confidence_fake,confidence_real  = DeepSupervision_acc2(accuracy_score, outputs, labels_data,features)
        
     
            count += 1
     
       
            data = [[fp, cf,cr] for fp, cf,cr in zip(img_name, confidence_fake,confidence_real)]
        
        
            labelcsv='test/wdf_wdf.csv'
            with open(labelcsv, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)

 

    f='test/wdf_wdf.csv'
    a='test/wdf_wdf-0.csv'
    b='test/wdf_wdf-1.csv'
    c='test/wdf_wdf-2.csv'
    string='real_'  #'real_'  #realff      real
    import csv

    filename =f

    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]

    header = ['img_name', 'fake_c', 'real_c']
    rows.insert(0, header)  
  

  


    output_filename = a
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows) 

    import pandas as pd

    df = pd.read_csv(a)

    avg_df = df.groupby("img_name")["real_c"].mean()

    new_df = avg_df.reset_index(name='avg_real_conf')

    new_df = new_df[["img_name", 'avg_real_conf']].rename(columns={'avg_real_conf': 'avg_real_conf'})

    new_df.to_csv(b, index=False)

    import pandas as pd

    df = pd.read_csv(b)

    df["label_realcon"] = (df["avg_real_conf"] >= 0.5).astype(int)

    df["label_ori"] = (df["img_name"].str.contains(string)).astype(int)

    new_df = df[["img_name", "avg_real_conf", "label_realcon", "label_ori"]].rename(columns={"avg_real_conf": "avg_real_conf"})

    df.to_csv(c, index=False)
    import numpy
    import pandas as pd
    from sklearn.metrics import roc_auc_score,average_precision_score
    from scipy.integrate import simps
    df = pd.read_csv(c)
    df['label_realcon'] = pd.to_numeric(df['label_realcon'], errors='coerce')
    df['label_ori'] = pd.to_numeric(df['label_ori'], errors='coerce')


    accuracy = sum(df['label_realcon'] == df['label_ori']) / len(df)
    print('Accuracy:', accuracy)

    # auc = roc_auc_score(df['label_ori'], df['label_realcon'])
    auc= roc_auc_score(df['label_ori'], df['avg_real_conf'])


    print('AUC:', auc)




    