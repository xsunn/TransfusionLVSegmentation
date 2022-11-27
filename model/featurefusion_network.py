import torch.nn.functional as F
from typing import Optional
from torch import nn,Tensor
import copy
import torch

# position encoding
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim):
        super(LearnedPositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_position_embeddings, embedding_dim))

    def forward(self, x):
        position_embeddings = self.position_embeddings
        return x + position_embeddings

#patch embedding
class PatchEmbedding(nn.Module):
    def __init__(self,imgSize,patchSize,in_channel,embeddingDim):
        super().__init__()

        img_size=(imgSize,imgSize)
        patch_size=(patchSize,patchSize)

        self.img_size=img_size
        self.patch_size=patch_size
        self.grid_size =(img_size[0]//patch_size[0],img_size[1]//patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]


        self.proj = nn.Conv2d(in_channel,embeddingDim,kernel_size=patchSize,stride=patchSize)

        self.norm = nn.LayerNorm(embeddingDim)
        self.max_position_embeddings=self.num_patches
        self.positionInfor=LearnedPositionalEncoding(max_position_embeddings=self.max_position_embeddings,embedding_dim=embeddingDim)

    def forward(self,x):
        # print("feature map size",x.shape)
        x=self.proj(x)
        x=x.flatten(2).transpose(1,2)
        x=self.norm(x)
        # print("embedding patch size",x.shape)
        x=x+self.positionInfor(x)
        return x

# self-attention attention
class ECA(nn.Module):
    def __init__(self,dim,heads=8,qkv_bias=False,qk_scale=None,dropout_rate=0.2):
        super().__init__()
        self.num_heads=heads
        head_dim  =dim//heads
        self.scale=qk_scale or head_dim **0.5
        self.dim=dim
        self.qkv=nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop=nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop =nn.Dropout(dropout_rate)
        self.norm=nn.LayerNorm(self.dim)

    def forward(self,x):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads, C// self.num_heads).permute(2,0,3,1,4)
        q,k,v=(qkv[0],qkv[1],qkv[2])

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x1=(attn @ v ).transpose(1,2).reshape(B,N,C)
        x1=self.proj(x1)
        x1=self.proj_drop(x1)
        x=x+x1
        x=self.norm(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )
    def forward(self, x):
        return self.net(x)

#cross-fusion attention
class CFA(nn.Module):
    def __init__(self,dim,hidden_dim,heads=8,qkv_bias=False,qk_scale=None,dropout_rate=0.2):
        super().__init__()
        self.num_heads=heads
        head_dim  =dim//heads
        self.scale=qk_scale or head_dim **0.5
        self.dim=dim
        self.q=nn.Linear(dim,dim,bias=qkv_bias)
        self.kv=nn.Linear(dim,dim*2,bias=qkv_bias)
        self.attn_drop=nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop =nn.Dropout(dropout_rate)

        self.FFN=FeedForward(dim,hidden_dim=hidden_dim,dropout_rate=dropout_rate)
        self.norm1=nn.LayerNorm(self.dim)
        self.norm2=nn.LayerNorm(self.dim)

    def forward(self,xq,xkv):
        B,N,C = xq.shape
        q = self.q(xq).reshape(B,N,self.num_heads, C// self.num_heads).permute(0,2,1,3)
        kv = self.kv(xkv).reshape(B,N,2,self.num_heads, C// self.num_heads).permute(2,0,3,1,4)
        q,k,v=(q,kv[0],kv[1])

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x1=(attn @ v ).transpose(1,2).reshape(B,N,C)
        x1=self.proj(x1)
        x1=self.proj_drop(x1)
        # x=x1
        x=xq+x1
        x=self.norm1(x)
        x2=self.FFN(x)
        x= x+x2
        x=self.norm2(x)
        return x

class CFAFusion(nn.Module):
    def __init__(self,dim,hidden_dim,heads=8,qkv_bias=False,qk_scale=None,dropout_rate=0.2):
        super().__init__()
        self.num_heads=heads
        head_dim  =dim//heads
        self.scale=qk_scale or head_dim **0.5
        self.dim=dim
        self.qkv=nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop=nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop =nn.Dropout(dropout_rate)
        self.FFN=FeedForward(dim,hidden_dim=hidden_dim,dropout_rate=dropout_rate)
        self.norm1=nn.LayerNorm(self.dim)
        self.norm2=nn.LayerNorm(self.dim)

    def forward(self,xq):
        B,N,C = xq.shape
        qkv = self.qkv(xq).reshape(B,N,3,self.num_heads, C// self.num_heads).permute(2,0,3,1,4)
        q,k,v=(qkv[0],qkv[1],qkv[1])
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=1)
        attn = self.attn_drop(attn)
        x1=(attn @ v ).transpose(1,2).reshape(B,N,C)
        x1=self.proj(x1)
        x1=self.proj_drop(x1)
        # x=x1
        x=xq+x1
        x=self.norm1(x)
        x2=self.FFN(x)
        x= x+x2
        x=self.norm2(x)
        return x


