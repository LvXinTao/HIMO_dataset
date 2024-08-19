import torch
from torch.utils.data._utils.collate import default_collate

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    
def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def himo_2o_collate_fn(batch):
    adapteed_batch=[
        {
            'inp':torch.tensor(b[6]).float(), # [nf,52*3+52*6+3+2*9]
            'length':b[7],
            'text':b[0],
            'obj1_bps':torch.tensor(b[1]).squeeze(0).float(), # [1024,3]
            'obj2_bps':torch.tensor(b[2]).squeeze(0).float(),
            'obj1_sampled_verts':torch.tensor(b[3]).float(), # [1024,3]
            'obj2_sampled_verts':torch.tensor(b[4]).float(),
            'init_state':torch.tensor(b[5]).float(), # [52*3+52*6+3+2*9]
            'betas':torch.tensor(b[8]).float(), # [10]
        } for b in batch
    ]
    inp_tensor=collate_tensors([b['inp'] for b in adapteed_batch]) # [bs,nf,52*3+52*6+3+2*9]
    len_batch=[b['length'] for b in adapteed_batch]
    len_tensor=torch.tensor(len_batch).long() # [B]
    mask_tensor=lengths_to_mask(len_tensor,inp_tensor.shape[1]).unsqueeze(1).unsqueeze(1) # [B,1,1,nf]

    text_batch=[b['text'] for b in adapteed_batch]
    o1b_tensor=torch.stack([b['obj1_bps'] for b in adapteed_batch],dim=0) # [B,1024,3]
    o2b_tensor=torch.stack([b['obj2_bps'] for b in adapteed_batch],dim=0) # [B,1024,3]
    o1sv_tensor=torch.stack([b['obj1_sampled_verts'] for b in adapteed_batch],dim=0) # [B,1024,3]
    o2sv_tensor=torch.stack([b['obj2_sampled_verts'] for b in adapteed_batch],dim=0) # [B,1024,3]
    init_state_tensor=torch.stack([b['init_state'] for b in adapteed_batch],dim=0) # [B,52*3+52*6+3+2*9]
    betas_tensor=torch.stack([b['betas'] for b in adapteed_batch],dim=0) # [B,10]

    cond={
        'y':{
            'mask':mask_tensor,
            'length':len_tensor,
            'text':text_batch,
            'obj1_bps':o1b_tensor,
            'obj2_bps':o2b_tensor,
            'obj1_sampled_verts':o1sv_tensor,
            'obj2_sampled_verts':o2sv_tensor,
            'init_state':init_state_tensor,
            'betas':betas_tensor
        }
    }
    return inp_tensor,cond

def himo_3o_collate_fn(batch):
    adapteed_batch=[
        {
            'inp':torch.tensor(b[8]).float(), # [nf,52*3+52*6+3+3*9]
            'length':b[9],
            'text':b[0],
            'obj1_bps':torch.tensor(b[1]).squeeze(0).float(), # [1024,3]
            'obj2_bps':torch.tensor(b[2]).squeeze(0).float(),
            'obj3_bps':torch.tensor(b[3]).squeeze(0).float(),
            'obj1_sampled_verts':torch.tensor(b[4]).float(), # [1024,3]
            'obj2_sampled_verts':torch.tensor(b[5]).float(),
            'obj3_sampled_verts':torch.tensor(b[6]).float(),
            'init_state':torch.tensor(b[7]).float(), # [52*3+52*6+3+3*9]
            'betas':torch.tensor(b[10]).float(), # [10]

        } for b in batch
    ]
    inp_tensor=collate_tensors([b['inp'] for b in adapteed_batch]) # [bs,nf,52*3+52*6+3+2*9]
    len_batch=[b['length'] for b in adapteed_batch]
    len_tensor=torch.tensor(len_batch).long() # [B]
    mask_tensor=lengths_to_mask(len_tensor,inp_tensor.shape[1]).unsqueeze(1).unsqueeze(1) # [B,1,1,nf]

    text_batch=[b['text'] for b in adapteed_batch]
    o1b_tensor=torch.stack([b['obj1_bps'] for b in adapteed_batch],dim=0) # [B,1024,3]
    o2b_tensor=torch.stack([b['obj2_bps'] for b in adapteed_batch],dim=0) # [B,1024,3]
    o3b_tensor=torch.stack([b['obj3_bps'] for b in adapteed_batch],dim=0) # [B,1024,3]
    o1sv_tensor=torch.stack([b['obj1_sampled_verts'] for b in adapteed_batch],dim=0) # [B,1024,3]
    o2sv_tensor=torch.stack([b['obj2_sampled_verts'] for b in adapteed_batch],dim=0) # [B,1024,3]
    o3sv_tensor=torch.stack([b['obj3_sampled_verts'] for b in adapteed_batch],dim=0) # [B,1024,3]
    init_state_tensor=torch.stack([b['init_state'] for b in adapteed_batch],dim=0) # [B,52*3+52*6+3+3*9]
    betas_tensor=torch.stack([b['betas'] for b in adapteed_batch],dim=0) # [B,10]

    cond={
        'y':{
            'mask':mask_tensor,
            'length':len_tensor,
            'text':text_batch,
            'obj1_bps':o1b_tensor,
            'obj2_bps':o2b_tensor,
            'obj3_bps':o3b_tensor,
            'obj1_sampled_verts':o1sv_tensor,
            'obj2_sampled_verts':o2sv_tensor,
            'obj3_sampled_verts':o3sv_tensor,
            'init_state':init_state_tensor,
            'betas':betas_tensor
        }
    }
    return inp_tensor,cond

def gt_collate_fn(batch):
    # sort batch by sent length
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)
    