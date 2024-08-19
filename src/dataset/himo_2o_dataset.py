import os
import os.path as osp
import torch
import h5py
from tqdm import tqdm
from torch.utils.data import Dataset
from src.utils.misc import to_tensor
import numpy as np

class HIMO_2O(Dataset):
    def __init__(self,
                args,
                split='train'):
        super(HIMO_2O,self).__init__()
        self.args=args
        self.split=split
        self.data_path=osp.join(args.data_dir,'processed_2o',f'{split}.h5')
        self.obj_bps_path=osp.join(args.data_dir,'processed_2o','object_bps.npz')
        self.obj_sampled_verts_path=osp.join(args.data_dir,'processed_2o','sampled_obj_verts.npz')
        self.max_frames=self.args.num_frames

        self.data=[]

        self.object_bps=dict(np.load(self.obj_bps_path,allow_pickle=True)) # 1,1024,3
        self.object_sampled_verts=dict(np.load(self.obj_sampled_verts_path,allow_pickle=True)) # 1024,3
        
        self.load_data()
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data=self.data[idx]
        m_len=data['obj1_state'].shape[0]

        full_pose=np.concatenate(
            [data['global_orient'],data['body_pose'],data['lhand_pose'],data['rhand_pose']],axis=1) # nf,52,6
        human_motion=np.concatenate(
            [
                data['smplx_joints'].reshape([m_len,-1]),
                full_pose.reshape([m_len,-1]),
                data['transl']
            ],axis=-1) # nf,52*3+52*6+3
        x_start=np.concatenate(
            [
                human_motion,
                data['obj1_state'],
                data['obj2_state']
            ],axis=-1) # nf,52*3+52*6+3+2*9
        text=data['text'].strip().split('#')[0] # caption
        if m_len<self.max_frames:
            x_start=np.concatenate([x_start,np.zeros([self.max_frames-m_len,x_start.shape[1]])],axis=0)
        obj1_bps=self.object_bps[data['obj1_name']]
        obj2_bps=self.object_bps[data['obj2_name']] # 1,1024,3
        obj1_sampled_verts=self.object_sampled_verts[data['obj1_name']]
        obj2_sampled_verts=self.object_sampled_verts[data['obj2_name']] # 1024,3

        init_state=x_start[0] # 52*3+52*6+3+2*9

        betas=data['betas'] # 10

        return text,obj1_bps,obj2_bps,obj1_sampled_verts,obj2_sampled_verts,init_state,x_start,m_len,betas

    def load_data(self):
        with h5py.File(self.data_path,'r') as f:
            for seq in tqdm(f.keys(),desc=f'Loading {self.split}:{self.args.dataset} dataset'):
                data={}
                for k in f[seq].keys():
                    if k!='object_state':
                        data[k]=f[seq][k][:]
                data['text']=f[seq]['text'][0].decode()
                data['global_orient']=f[seq]['global_orient'][:][:,None,:] # nf,1,6

                obj1,obj2=sorted(f[seq]['object_state'].keys())
                data['obj1_name']=obj1
                data['obj2_name']=obj2
                data['obj1_state']=f[seq]['object_state'][obj1][:] # nframes,9
                data['obj2_state']=f[seq]['object_state'][obj2][:] # nframes,9
                self.data.append(data)
    


                