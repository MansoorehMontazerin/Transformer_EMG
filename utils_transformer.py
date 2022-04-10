import torch
from Prep_func_transformer import loaddata_filt
import numpy as np


def load_dataset(window_size,skip_step,idx):
    """
    This function calls the loaddata function which loads train and test data from data folders.
    
    Parameters
    ----------
    
     idx:
         Subject index
         
     """
    
    tr_sam=[]
    tr_class=[]
    ts_sam=[]
    ts_class=[]
    
    tr_sam,tr_class,_=loaddata_filt('train',window_size,skip_step,idx)
    ts_sam,ts_class,_=loaddata_filt('test',window_size,skip_step,idx)
    
    return tr_sam,tr_class,ts_sam,ts_class




def init_dataset(opt):
   
    """
   This function loads data from data folders, converts them to torch tensors
   and creates data loaders for train and test data.
        
    """
    
    tr_sam=[]
    tr_class=[]
    ts_sam=[]
    ts_class=[]                                          
    tr_sam1=[]
    tr_class1=[]
    ts_sam1=[]
    ts_class1=[]    
    tr_sam2=[]
    tr_class2=[]
    ts_sam2=[]
    ts_class2=[]
    
    tr_sam,tr_class,ts_sam,ts_class=load_dataset(64,8000,opt.sub_idx)
    
    
    tr_sam1  = np.array(tr_sam)
    tr_class1 = np.array(tr_class)
    tr_sam2  = (torch.Tensor(tr_sam1))
    tr_class2 = torch.IntTensor(tr_class1)
    
    train_dataset =torch.utils.data.TensorDataset(tr_sam2,tr_class2)
    tr_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=opt.batch_size,
                                                shuffle=opt.shuffle,
                                                num_workers=opt.num_workers)
    
    ts_sam1  = np.array(ts_sam)
    ts_class1 = np.array(ts_class)
    ts_sam2  = (torch.Tensor(ts_sam1))
    ts_class2 = torch.IntTensor(ts_class1)
    
    test_dataset =torch.utils.data.TensorDataset(ts_sam2,ts_class2)
    ts_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=opt.batch_size,
                                                shuffle=opt.shuffle,
                                                num_workers=opt.num_workers)
    
    return tr_dataloader,ts_dataloader




