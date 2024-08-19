from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
import numpy as np
from loguru import logger
import os.path as osp

class decomp_dataset(Dataset):
    def __init__(self,opt,split='train'):
        self.opt=opt
        
        self.data=[]
        self.lengths=[]
        self.split=split
        self.load_data()

        self.cumsum = np.cumsum([0] + self.lengths)
        logger.info("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def __len__(self):
        return self.cumsum[-1]
    
    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx+self.opt.window_size]
        return motion
    
    def load_data(self):
        with h5py.File(osp.join(self.opt.data_root,"processed_%s"%self.opt.mode,"{}.h5".format(self.split)),'r') as f:
            for seq in tqdm(f.keys(),desc=f'Loading {self.split} dataset'):
                
                body_pose=f[seq]['body_pose'][:] # nf,21,6
                global_orient=f[seq]['global_orient'][:][:,None,:] # nf,1,6
                lhand_pose=f[seq]['lhand_pose'][:] # nf,15,6
                rhand_pose=f[seq]['rhand_pose'][:] # nf,15,6
                transl=f[seq]['transl'][:] # nf,3
                smplx_joints=f[seq]['smplx_joints'][:]
                smplx_joints=smplx_joints.reshape(smplx_joints.shape[0],-1) # nf,52*3
                full_pose=np.concatenate([global_orient,body_pose,lhand_pose,rhand_pose],axis=1)
                full_pose=full_pose.reshape(full_pose.shape[0],-1) # nf,52*6
                human_motion=np.concatenate([smplx_joints,full_pose,transl],axis=-1) # nf,52*6+52*3+3
                if self.opt.mode=='2o':
                    o1,o2=sorted(f[seq]['object_state'].keys())
                    o1_state=f[seq]['object_state'][o1][:]
                    o2_state=f[seq]['object_state'][o2][:]
                    object_state=np.concatenate([o1_state,o2_state],axis=-1)
                elif self.opt.mode=='3o':
                    o1,o2,o3=sorted(f[seq]['object_state'].keys())
                    o1_state=f[seq]['object_state'][o1][:]
                    o2_state=f[seq]['object_state'][o2][:]
                    o3_state=f[seq]['object_state'][o3][:]
                    object_state=np.concatenate([o1_state,o2_state,o3_state],axis=-1)
                motion=np.concatenate([human_motion,object_state],axis=-1) # nf,(52*6+52*3+3)+(2*(6+3))
                if motion.shape[0]<self.opt.window_size:
                    continue
                self.lengths.append(motion.shape[0]-self.opt.window_size)
                self.data.append(motion)