
# coding: utf-8

# In[1]:


get_ipython().system('pip install fastai==0.7.0')
get_ipython().system('pip install torch==0.4.0')
get_ipython().system('pip install torchtext==0.2.3')


# In[2]:


from fastai.conv_learner import *
from fastai.dataset import *

import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split


# In[3]:


PATH = './'
TRAIN = '../input/inria-aerial-labeling-dataset/images/images/'
#TEST = '../input/airbus-ship-detection/test/'
SEGMENTATION = '../input/inria-aerial-labeling-dataset/gt/gt/'
#PRETRAINED = '../input/fine-tuning-resnet34-on-ship-detection-384/models/Resnet34_lable_384_1.h5'


# In[4]:


nw = 2   #number of workers for data loader
arch = resnet34 #specify target architecture


# In[5]:


train_names = [f for f in os.listdir(TRAIN)]
#test_names = [f for f in os.listdir(TEST)]
#5% of data in the validation set is sufficient for model evaluation
train_names, test_names = train_test_split(train_names, test_size=0.1, random_state=42)
tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)
#segmentation_df = pd.read_csv(os.path.join(PATH, SEGMENTATION)).set_index('ImageId')


# In[6]:


class RandomLighting(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b,self.c = b,c

    def set_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        if is_y and self.tfm_y != TfmType.PIXEL: return x  #add this line to fix the bug
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1/(c-1) if c<0 else c+1
        x = lighting(x, b, c)
        return x


# In[7]:


class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        super().__init__(fnames, transform, path)
    
    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        if self.sz == 5000: return img 
        else: return cv2.resize(img, (self.sz, self.sz))
    
    def get_y(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        if self.sz == 5000: return img 
        else: return cv2.resize(img, (self.sz, self.sz))
    
    def get_c(self): return 0


# In[8]:


def get_data(sz,bs):
    #data augmentation
    aug_tfms = [RandomRotate(20, tfm_y=TfmType.CLASS),
                RandomDihedral(tfm_y=TfmType.CLASS),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)]
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, 
                aug_tfms=aug_tfms)
    tr_names = tr_n if (len(tr_n)%bs == 0) else tr_n[:-(len(tr_n)%bs)] #cut incomplete batch
    ds = ImageData.get_ds(pdFilesDataset, (tr_names,TRAIN), 
                (val_n,TRAIN), tfms, test=(test_names,TRAIN))
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    #md.is_multi = False
    return md


# In[9]:


cut,lr_cut = model_meta[arch]


# In[10]:


def get_base():                   #load ResNet34 model
    layers = cut_model(arch(True), cut)
    return nn.Sequential(*layers)

def load_pretrained(model, path): #load a model pretrained on ship/no-ship classification
    weights = torch.load(PRETRAINED, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights, strict=False)
            
    return model


# In[11]:


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bn(F.relu(cat_p))

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()
    
class Unet34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,256)
        self.up2 = UnetBlock(256,128,256)
        self.up3 = UnetBlock(256,64,256)
        self.up4 = UnetBlock(256,64,256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)
        
    def forward(self,x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return x[:,0]
    
    def close(self):
        for sf in self.sfs: sf.remove()
            
class UnetModel():
    def __init__(self,model,name='Unet'):
        self.model,self.name = model,name
        
    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model)[1:]]


# In[12]:


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


# In[13]:


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val +             ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()


# In[14]:


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        
    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()


# In[15]:


def dice(pred, targs):
    pred = (pred>0).float()
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)

def IoU(pred, targs):
    pred = (pred>0).float()
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)


# In[16]:


m_base = get_base()
m = to_gpu(Unet34(m_base))
models = UnetModel(m)


# In[17]:


models.model


# In[18]:


sz = 2048 #image size
bs = 1  #batch size

md = get_data(sz,bs)


# In[19]:


learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam
learn.crit = MixedLoss(10.0, 2.0)
learn.metrics=[accuracy_thresh(0.5),dice,IoU]
wd=1e-7
lr = 1e-2


# In[20]:


learn.freeze_to(1)


# In[21]:


learn.fit(lr,1,wds=wd,cycle_len=1,use_clr=(5,8))


# In[22]:


#learn.save('Unet34_256_0')


# In[23]:


lrs = np.array([lr/100,lr/10,lr])
learn.unfreeze() #unfreeze the encoder
learn.bn_freeze(True)


# In[24]:


learn.fit(lrs,5,wds=wd,cycle_len=1,use_clr=(20,8))


# In[25]:


learn.sched.plot_lr()


# In[26]:


#learn.save('Unet34_256_1')


# In[27]:


def Show_images(x,yp,yt):
    columns = 3
    rows = min(bs,8)
    fig=plt.figure(figsize=(columns*4, rows*4))
    for i in range(rows):
        fig.add_subplot(rows, columns, 3*i+1)
        plt.axis('off')
        plt.imshow(x[i])
        fig.add_subplot(rows, columns, 3*i+2)
        plt.axis('off')
        plt.imshow(yp[i])
        fig.add_subplot(rows, columns, 3*i+3)
        plt.axis('off')
        plt.imshow(yt[i])
    plt.show()


# In[28]:


learn.model.eval();
x,y = next(iter(md.val_dl))
yp = to_np(F.sigmoid(learn.model(V(x))))


# In[29]:


Show_images(np.asarray(md.val_ds.denorm(x)), yp, y)


# In[30]:


sz = 1024 #image size
bs = 10  #batch size

md = get_data(sz,bs)
learn.set_data(md)
learn.unfreeze()
learn.bn_freeze(True)


# In[31]:


learn.fit(lrs/5,1,wds=wd,cycle_len=1,use_clr=(10,8))


# In[32]:


learn.save('Unet34_384_1')


# In[33]:


learn.model.eval();
x,y = next(iter(md.val_dl))
yp = to_np(F.sigmoid(learn.model(V(x))))


# In[34]:


Show_images(np.asarray(md.val_ds.denorm(x)), yp, y)


# In[35]:


import gc
gc.collect()


# In[36]:


sz = 512 #image size
bs = 6  #batch size

md = get_data(sz,bs)
learn.set_data(md)
learn.unfreeze()
learn.bn_freeze(True)


# In[37]:


learn.fit(lrs/10,1,wds=wd,cycle_len=1,use_clr=(10,8))


# In[38]:


sz = 768 #image size
bs = 1 #batch size

md = get_data(sz,bs)
learn.set_data(md)
learn.unfreeze()
learn.bn_freeze(True)

learn.fit(lrs/10,1,wds=wd,cycle_len=1,use_clr=(10,8))


# In[39]:


sz = 1024 #image size
bs = 1 #batch size

md = get_data(sz,bs)
learn.set_data(md)
learn.unfreeze()
learn.bn_freeze(True)

learn.fit(lrs/10,1,wds=wd,cycle_len=1,use_clr=(10,8))


# In[40]:


sz = 2048 #image size
bs = 1 #batch size

md = get_data(sz,bs)
learn.set_data(md)
learn.unfreeze()
learn.bn_freeze(True)

learn.fit(lrs/10,1,wds=wd,cycle_len=1,use_clr=(10,8))


# In[41]:


sz = 3076 #image size
bs = 1 #batch size

md = get_data(sz,bs)
learn.set_data(md)
learn.unfreeze()
learn.bn_freeze(True)

learn.fit(lrs/10,1,wds=wd,cycle_len=1,use_clr=(10,8))

