import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger
import clip
from src.model.blocks import TransformerBlock

class NET_3O(nn.Module):
    def __init__(self,
                 n_feats=(52*3+52*6+3+3*9),
                 clip_dim=512,
                 latent_dim=512,
                 ff_size=1024, 
                 num_layers=8, 
                 num_heads=4, 
                 dropout=0.1,
                 ablation=None, 
                 activation="gelu",**kwargs):
        super().__init__()

        self.n_feats = n_feats
        self.clip_dim = clip_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation

        self.cond_mask_prob=kwargs.get('cond_mask_prob',0.1)

        # clip and text embedder
        self.embed_text=nn.Linear(self.clip_dim,self.latent_dim)
        self.clip_version='ViT-B/32'
        self.clip_model=self.load_and_freeze_clip(self.clip_version)

        # object_geometry embedder
        self.embed_obj_bps=nn.Linear(1024*3,self.latent_dim)
        # object init state embeddr
        self.embed_obj_pose=nn.Linear(3*self.latent_dim+3*9+3*9,self.latent_dim)

        # human embedder
        self.embed_human_pose=nn.Linear(2*(52*3+52*6+3),self.latent_dim)

        # position encoding
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=self.dropout)

        # TODO:unshared transformer layers for human and objects,they fuse feature in the middle layers
        seqTransEncoderLayer=nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        # # object transformer layers
        # self.object_trans_encoder=nn.TransformerEncoder(seqTransEncoderLayer,num_layers=self.num_layers)
        # # human transformer encoder
        # self.human_trans_encoder=nn.TransformerEncoder(seqTransEncoderLayer,num_layers=self.num_layers)

        # # Mutal cross attention
        # self.communication_module=nn.ModuleList()
        # for i in range(8):
        #     self.communication_module.append(MutalCrossAttentionBlock(self.latent_dim,self.num_heads,self.ff_size,self.dropout))
        self.obj_blocks=nn.ModuleList()
        self.human_blocks=nn.ModuleList()
        for i in range(self.num_layers):
            self.obj_blocks.append(TransformerBlock(self.latent_dim, self.num_heads, self.ff_size, self.dropout, self.activation))
            self.human_blocks.append(TransformerBlock(self.latent_dim, self.num_heads, self.ff_size, self.dropout, self.activation))

        
        # embed the timestep
        self.embed_timestep=TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # object output process
        self.obj_output_process=nn.Linear(self.latent_dim,3*9)
        # human motion output process
        self.human_output_process=nn.Linear(self.latent_dim,52*3+52*6+3)

    def forward(self,x,timesteps,y=None):
        bs,nframes,n_feats=x.shape
        emb=self.embed_timestep(timesteps) # [1, bs, latent_dim]

        enc_text=self.encode_text(y['text'])
        emb+=self.mask_cond(enc_text) # 1,bs,latent_dim

        x=x.permute((1,0,2)) # nframes,bs,nfeats
        human_x,obj_x=torch.split(x,[52*3+52*6+3,3*9],dim=-1) # [nframes,bs,52*3+52*6+3],[nframes,bs,3*9]

        # encode object geometry
        obj1_bps,obj2_bps,obj3_bps=y['obj1_bps'].reshape(bs,-1),y['obj2_bps'].reshape(bs,-1),y['obj3_bps'].reshape(bs,-1) # [b,1024,3]
        obj1_bps_emb=self.embed_obj_bps(obj1_bps) # [b,latent_dim]
        obj2_bps_emb=self.embed_obj_bps(obj2_bps) # [b,latent_dim]
        obj3_bps_emb=self.embed_obj_bps(obj3_bps) # [b,latent_dim]
        obj_geo_emb=torch.concat([obj1_bps_emb,obj2_bps_emb,obj3_bps_emb],dim=-1).unsqueeze(0).repeat((nframes,1,1)) # [nf,b,3*latent_dim]

        # init_state,mask the other frames by padding zeros
        init_state=y['init_state'].unsqueeze(0) # [1,b,52*3+52*6+3+3*9]
        padded_zeros=torch.zeros((nframes-1,bs,52*3+52*6+3+3*9),device=init_state.device)
        init_state=torch.concat([init_state,padded_zeros],dim=0) # [nf,b,52*3+52*6+3+3*9]

        # seperate the object and human init state
        human_init_state=init_state[:,:,:52*3+52*6+3] # [nf,b,52*3+52*6+3]
        obj_init_state=init_state[:,:,52*3+52*6+3:] # [nf,b,3*9]

        # Object branch
        obj_emb=self.embed_obj_pose(torch.concat([obj_geo_emb,obj_init_state,obj_x],dim=-1)) # nframes,bs,latent_dim
        obj_seq_prev=self.sequence_pos_encoder(obj_emb) # [nf,bs,latent_dim]


        # Human branch
        human_emb=self.embed_human_pose(torch.concat([human_init_state,human_x],dim=-1)) # nframes,bs,latent_dim
        human_seq_prev=self.sequence_pos_encoder(human_emb) # [nf,bs,latent_dim]

        mask=y['mask'].squeeze(1).squeeze(1) # [bs,nf]
        key_padding_mask=~mask # [bs,nf]

        obj_seq_prev=obj_seq_prev.permute((1,0,2)) # [bs,nf,latent_dim]
        human_seq_prev=human_seq_prev.permute((1,0,2)) # [bs,nf,latent_dim]
        emb=emb.squeeze(0) # [bs,latent_dim]
        for i in range(self.num_layers):
            obj_seq=self.obj_blocks[i](obj_seq_prev,human_seq_prev,emb, key_padding_mask=key_padding_mask)
            human_seq=self.human_blocks[i](human_seq_prev,obj_seq_prev,emb, key_padding_mask=key_padding_mask)
            obj_seq_prev=obj_seq
            human_seq_prev=human_seq
        obj_seq=obj_seq.permute((1,0,2)) # [nf,bs,latent_dim]
        human_seq=human_seq.permute((1,0,2))
        

        obj_output=self.obj_output_process(obj_seq) # [nf,bs,3*9]
        human_output=self.human_output_process(human_seq) # [nf,bs,52*3+52*6+3]

        output=torch.concat([human_output,obj_output],dim=-1) # [nf,bs,52*3+52*6+3+3*9]

        return output.permute((1,0,2))

    def encode_text(self,raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 40
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
