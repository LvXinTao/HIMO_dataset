import os
import os.path as osp
from typing import Any
import numpy as np
import h5py
from tqdm import tqdm
from torch.utils.data import  Dataset
import codecs as cs
from torch.utils.data._utils.collate import default_collate

class feature_extractor_dataset(Dataset):
    def __init__(self,args,split,w_vectorizer):
        self.args=args
        self.w_vectorizer=w_vectorizer
        self.max_motion_length=args.max_motion_length
        self.split=split
        self.data_path=osp.join(args.data_root,'processed_{}'.format(args.mode),f'{split}.h5')

        self.data=[]
        self.load_data()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data=self.data[idx]
        motion,m_length,text_list=data['motion'],data['length'],data['text']
        caption,tokens=text_list['caption'],text_list['tokens']

        if len(tokens) < self.args.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.args.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.args.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length<self.max_motion_length:
            # padding
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        
        return word_embeddings,pos_one_hots,caption,sent_len,motion,m_length,'_'.join(tokens)

    def load_data(self):
        with h5py.File(self.data_path,'r') as f:
            for seq in tqdm(f.keys(),desc=f'Loading {self.split}:{self.args.dataset} dataset'):
                data={}
                text=f[seq]['text'][0].decode()
    
                text_dict={}
                line_split = text.strip().split('#')
                caption = line_split[0]
                tokens = line_split[1].split(' ')

                text_dict['caption'] = caption
                text_dict['tokens'] = tokens
                
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
                if self.args.mode=='2o':
                    o1,o2=sorted(f[seq]['object_state'].keys())
                    o1_state=f[seq]['object_state'][o1][:]
                    o2_state=f[seq]['object_state'][o2][:]
                    object_state=np.concatenate([o1_state,o2_state],axis=-1)
                elif self.args.mode=='3o':
                    o1,o2,o3=sorted(f[seq]['object_state'].keys())
                    o1_state=f[seq]['object_state'][o1][:]
                    o2_state=f[seq]['object_state'][o2][:]
                    o3_state=f[seq]['object_state'][o3][:]
                    object_state=np.concatenate([o1_state,o2_state,o3_state],axis=-1)
                motion=np.concatenate([human_motion,object_state],axis=-1) # nf,(52*6+52*3+3)+(2*(6+3))

                data['motion']=motion
                data['length']=data['motion'].shape[0]
                data['text']=text_dict
                self.data.append(data)

def collate_fn(batch):
    # sort batch by sent length
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)