from src.model.layers import *

class TransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.,
                 cond_abl=False,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, y, emb=None, key_padding_mask=None):
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block(h1, y, emb, key_padding_mask)
        h2 = h2 + h1
        out = self.ffn(h2, emb)
        out = out + h2
        return out


class MutalCrossAttentionBlock(nn.Module):
    def __init__(
            self,
            latent_dim=512,
            num_heads=8,
            ff_size=1024,
            dropout=0
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout

        self.object_linear=nn.Linear(self.latent_dim,self.latent_dim)
        self.human_linear=nn.Linear(self.latent_dim,self.latent_dim)

        self.ca_block=CrossAttention(self.latent_dim,self.num_heads)
        
        self.object_ffn=nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.ff_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.ff_size, self.latent_dim),
            nn.Dropout(self.dropout)
        )
        self.human_ffn=nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.ff_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.ff_size, self.latent_dim),
            nn.Dropout(self.dropout)
        )
    
    def forward(self,object_feature,human_feature,key_padding_mask=None):
        """
        object_feature: N,B,L
        human_feature: N,B,L
        """
        object_feature=object_feature.permute(1,0,2)
        human_feature=human_feature.permute(1,0,2)

        object_feature=self.object_linear(object_feature)
        human_feature=self.human_linear(human_feature)

        object_feature=self.ca_block(object_feature,human_feature,human_feature,key_padding_mask)
        object_feature=self.object_ffn(object_feature)

        human_feature=self.ca_block(human_feature,object_feature,object_feature,key_padding_mask)
        human_feature=self.human_ffn(human_feature)

        object_feature=object_feature.permute(1,0,2)
        human_feature=human_feature.permute(1,0,2)

        return object_feature,human_feature