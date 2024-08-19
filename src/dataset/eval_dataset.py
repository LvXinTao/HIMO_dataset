import os
import os.path as osp
import numpy as np
import h5py
from tqdm import tqdm
from torch.utils.data import  Dataset
import codecs as cs
from torch.utils.data._utils.collate import default_collate
from src.utils.word_vectorizer import WordVectorizer

class Evaluation_Dataset(Dataset):
    def __init__(self,args,split='test',mode='gt'):
        self.args=args
        self.max_motion_length=self.args.max_motion_length
        self.split=split
        self.mode=mode
        self.obj=self.args.obj
        self.data_path=osp.join(args.data_dir,'processed_{}'.format(self.obj),f'{split}.h5')
        self.data_path=osp.join(args.data_dir,'processed_{}'.format(self.obj),f'{split}.h5')
        self.obj_bps_path=osp.join(args.data_dir,'processed_{}'.format(self.obj),'object_bps.npz')
        self.obj_sampled_verts_path=osp.join(args.data_dir,'processed_{}'.format(self.obj),'sampled_obj_verts.npz')
        self.w_vectorizer=WordVectorizer(osp.join(self.args.data_dir,'glove'),'himo_vab')

        self.data=[]
        self.load_data()

        self.object_bps=dict(np.load(self.obj_bps_path,allow_pickle=True)) # 1,1024,3
        self.object_sampled_verts=dict(np.load(self.obj_sampled_verts_path,allow_pickle=True)) # 1024,3

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
                
        if self.mode=='gt':
            return word_embeddings,pos_one_hots,caption,sent_len,motion,m_length,'_'.join(tokens)
        elif self.mode=='eval':
            if self.args.obj=='2o':
                obj1_bps=self.object_bps[data['obj1_name']].squeeze(0) # 1024,3
                obj2_bps=self.object_bps[data['obj2_name']].squeeze(0) # 1024,3
                init_state=motion[0] # 52*3+52*6+3+2*9
                return word_embeddings,pos_one_hots,caption,sent_len,motion,m_length,'_'.join(tokens),\
                    obj1_bps,obj2_bps,init_state.astype(np.float32),data['obj1_name'],data['obj2_name'],data['betas']
            elif self.args.obj=='3o':
                obj1_bps=self.object_bps[data['obj1_name']].squeeze(0)
                obj2_bps=self.object_bps[data['obj2_name']].squeeze(0)
                obj3_bps=self.object_bps[data['obj3_name']].squeeze(0)
                init_state=motion[0]
                return word_embeddings,pos_one_hots,caption,sent_len,motion,m_length,'_'.join(tokens),\
                    obj1_bps,obj2_bps,obj3_bps,init_state.astype(np.float32),data['obj1_name'],data['obj2_name'],data['obj3_name'],data['betas']


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
                
                betas=f[seq]['betas'][:] # nf,10
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
                if self.args.obj=='2o':
                    o1,o2=sorted(f[seq]['object_state'].keys())
                    data['obj1_name']=o1
                    data['obj2_name']=o2
                    o1_state=f[seq]['object_state'][o1][:]
                    o2_state=f[seq]['object_state'][o2][:]
                    object_state=np.concatenate([o1_state,o2_state],axis=-1)
                elif self.args.obj=='3o':
                    o1,o2,o3=sorted(f[seq]['object_state'].keys())
                    data['obj1_name']=o1
                    data['obj2_name']=o2
                    data['obj3_name']=o3
                    o1_state=f[seq]['object_state'][o1][:]
                    o2_state=f[seq]['object_state'][o2][:]
                    o3_state=f[seq]['object_state'][o3][:]
                    object_state=np.concatenate([o1_state,o2_state,o3_state],axis=-1)
                motion=np.concatenate([human_motion,object_state],axis=-1) # nf,(52*6+52*3+3)+(2*(6+3))

                data['motion']=motion
                data['length']=data['motion'].shape[0]
                data['text']=text_dict
                data['betas']=betas
                self.data.append(data)   