import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from tqdm import tqdm
from bps_torch.bps import bps_torch
import os.path as osp
import os
from src.utils.rotation_conversion import sixd2rot_torch
from src.dataset.tensors import gt_collate_fn
from src.utils import dist_utils
from src.dataset.tensors import lengths_to_mask

def get_eval_gen_loader(args,model,diffusion,gen_loader,
                    max_motion_length,batch_size,
                    mm_num_samples,mm_num_repeats,num_samples_limit,scale):
    dataset=Evaluation_generator_Dataset(args,model,diffusion,gen_loader,
                                         max_motion_length,mm_num_samples,mm_num_repeats,num_samples_limit,scale)
     
    mm_dataset=MM_generator_Dataset('test',dataset,gen_loader.dataset.w_vectorizer)

    motion_loader=DataLoader(dataset,batch_size=batch_size,num_workers=4,drop_last=True,collate_fn=gt_collate_fn)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')
    return motion_loader,mm_motion_loader

class MM_generator_Dataset(Dataset):
    def __init__(self, opt, motion_dataset, w_vectorizer):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        m_lens = []
        motions = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            # We don't need the following logic because our sample func generates the full tensor anyway:
            # if len(motion) < self.opt.max_motion_length:
            #     motion = np.concatenate([motion,
            #                              np.zeros((self.opt.max_motion_length - len(motion), motion.shape[1]))
            #                              ], axis=0)
            motion = motion[None, :]
            motions.append(motion)
        m_lens = np.array(m_lens, dtype=np.int)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        # print(m_lens)
        # print(sort_indx)
        # print(m_lens[sort_indx])
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        return motions, m_lens
    
class Evaluation_generator_Dataset(Dataset):
    def __init__(self,args,model,diffusion,gen_loader,
                 max_motion_length,mm_num_samples,mm_num_repeats,num_samples_limit,scale=1.0):
        self.dataloader=gen_loader
        assert mm_num_samples<len(self.dataloader.dataset)

        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length
        model_sample_fn=(
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )
        
        real_num_batches = len(self.dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // self.dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion=[]
        mm_generated_motion=[]
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // self.dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()

        with torch.no_grad():
            for i,eval_data_batch in tqdm(enumerate(self.dataloader),total=len(self.dataloader)):

                # if i==1:
                #     break
                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break
                    
                if args.obj=='2o':
                    word_embeddings,pos_one_hots,caption,sent_len,motion,m_length,tokens,\
                        obj1_bps,obj2_bps,init_state,obj1_name,obj2_name,betas=eval_data_batch
                    tokens=[t.split('_') for t in tokens]

                    model_kwargs={
                                'y':{
                                    'length':m_length.to(dist_utils.dev()), # [bs]
                                    'text':caption,
                                    'obj1_bps':obj1_bps.to(dist_utils.dev()),
                                    'obj2_bps':obj2_bps.to(dist_utils.dev()),
                                    'init_state':init_state.to(dist_utils.dev()),
                                    'mask':lengths_to_mask(m_length,motion.shape[1]).unsqueeze(1).unsqueeze(1).to(dist_utils.dev()) # [bs,1,1,nf]
                                }
                            }
                elif args.obj=='3o':
                    word_embeddings,pos_one_hots,caption,sent_len,motion,m_length,tokens,\
                        obj1_bps,obj2_bps,obj3_bps,init_state,obj1_name,obj2_name,obj3_name,betas=eval_data_batch
                    tokens=[t.split('_') for t in tokens]

                    model_kwargs={
                                'y':{
                                    'length':m_length.to(dist_utils.dev()),
                                    'text':caption,
                                    'obj1_bps':obj1_bps.to(dist_utils.dev()),
                                    'obj2_bps':obj2_bps.to(dist_utils.dev()),
                                    'obj3_bps':obj3_bps.to(dist_utils.dev()),
                                    'init_state':init_state.to(dist_utils.dev()),
                                    'mask':lengths_to_mask(m_length,motion.shape[1]).unsqueeze(1).unsqueeze(1).to(dist_utils.dev()) # [bs,1,1,nf]
                                }
                            }

                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_utils.dev()) * scale

                mm_num_now = len(mm_generated_motion) // self.dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    
                    model_out_sample=model_sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                    ) # bs,nf,315

                    # export the model result 
                    network=args.model_path.split('/')[-2]
                    export_path=osp.join('./export_results',network,'{}.npz'.format(i))
                    os.makedirs(osp.dirname(export_path),exist_ok=True)
                    if args.obj=='2o':
                        output_dict={str(bs_i):
                            {
                                'out':model_out_sample[bs_i].squeeze().cpu().numpy(),
                                'length':m_length[bs_i].cpu().numpy(),
                                'caption':caption[bs_i],
                                'obj1_name':obj1_name[bs_i],
                                'obj2_name':obj2_name[bs_i],
                                'betas':betas[bs_i].cpu().numpy(),
                            } for bs_i in range(self.dataloader.batch_size)
                        }
                    elif args.obj=='3o':
                        output_dict={str(bs_i):
                            {
                                'out':model_out_sample[bs_i].squeeze().cpu().numpy(),
                                'length':m_length[bs_i].cpu().numpy(),
                                'caption':caption[bs_i],
                                'obj1_name':obj1_name[bs_i],
                                'obj2_name':obj2_name[bs_i],
                                'obj3_name':obj3_name[bs_i],
                                'betas':betas[bs_i].cpu().numpy(),
                            } for bs_i in range(self.dataloader.batch_size)
                        }
                    np.savez(export_path,**output_dict)

                    if t==0:
                        sub_dicts=[
                            {
                                'motion':model_out_sample[bs_i].squeeze().cpu().numpy(),
                                'length':m_length[bs_i].cpu().numpy(),
                                'caption':caption[bs_i],
                                'tokens':tokens[bs_i],
                                'cap_len':sent_len[bs_i].cpu().numpy(),
                            } for bs_i in range(self.dataloader.batch_size)
                        ]
                        generated_motion+=sub_dicts
                    if is_mm:
                        mm_motions+=[
                            {
                                'motion':model_out_sample[bs_i].squeeze().cpu().numpy(),
                                'length':m_length[bs_i].cpu().numpy(),
                            } for bs_i in range(self.dataloader.batch_size)
                        ]
                if is_mm:
                    mm_generated_motion+=[
                        {
                            'caption':model_kwargs['y']['text'][bs_i],
                            'tokens':tokens[bs_i],
                            'cap_len':len(tokens[bs_i]),
                            'mm_motions':mm_motions[bs_i::self.dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                        } for bs_i in range(self.dataloader.batch_size)
                    ]
        self.generated_motion=generated_motion
        self.mm_generated_motion=mm_generated_motion
        self.w_vectorizer=self.dataloader.dataset.w_vectorizer

    def __len__(self):
        return len(self.generated_motion)
    
    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)