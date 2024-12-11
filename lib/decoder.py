
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
import math
from timm.models.layers import DropPath, trunc_normal_

from torch.nn import functional as F


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, v_dim=512, l_dim=768, hidden_dim=512, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sra = False, pool_ratio=2):
        super().__init__()

        self.dim = hidden_dim
        # self.dim2 = dim2
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(v_dim, hidden_dim, bias=qkv_bias)
        self.kv = nn.Linear(l_dim, hidden_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_dim, v_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sra = sra
        if sra:
            self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
            self.sr = nn.Conv2d(l_dim, l_dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(l_dim)
            self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        

    def forward(self, x, l, l_mask, h=None, w=None, visual=False):
        B, N, C = x.shape
        _, L, _ = l.shape
        if self.sra:
            x_ = l.permute(0, 2, 1).reshape(B, C, h, w)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            l = self.act(x_)
            _, L, _ = l.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(l).reshape(B, L, 2, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if l_mask is not None:
            attn = attn + l_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    
class Block(nn.Module):

    def __init__(self, v_dim, l_dim,  hidden_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sra = False, pool_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(v_dim)
        self.norm2 = norm_layer(l_dim)
        self.norm3 = norm_layer(v_dim)
        
        self.attn = CrossAttention(v_dim, l_dim, hidden_dim, num_heads=num_heads, sra = sra, pool_ratio=pool_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(v_dim * mlp_ratio)
        self.mlp = Mlp(in_features=v_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, l, l_mask, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(l), l_mask, H, W))
        x = x + self.drop_path(self.mlp(self.norm3(x), H, W))

        return x
    
class QLBlock(nn.Module):

    def __init__(self, q_dim, k_dim, hidden_dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(q_dim)
        self.norm2 = norm_layer(k_dim)
        self.norm3 = norm_layer(q_dim)
        
        self.attn = CrossAttention(q_dim, k_dim, hidden_dim, num_heads=num_heads)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(q_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(q_dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, q_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        

    def forward(self, q, kv, mask):
     
        q = q + self.drop_path(self.attn(self.norm1(q), self.norm2(kv), mask))
        q = q + self.drop_path(self.mlp(self.norm3(q)))

        return q

class TopKAttention(nn.Module):
    def __init__(self, v_dim=512, l_dim=768, hidden_dim=512, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = hidden_dim
        # self.dim2 = dim2
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(v_dim, hidden_dim, bias=qkv_bias)
        self.kv = nn.Linear(l_dim, hidden_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_dim, v_dim)
        self.proj_drop = nn.Dropout(proj_drop)       

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        

    def forward(self, x, vfeats, topk_mask):
        B, L, C = x.shape
        _, N, _ = vfeats.shape
        
        q = self.q(x).reshape(B, L, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(vfeats).reshape(B, N, 2, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if topk_mask is not None:
            attn = attn + topk_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, L, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x 

class TopKBlock(nn.Module):

    def __init__(self, q_dim, k_dim, hidden_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(q_dim)
        self.norm2 = norm_layer(k_dim)
        self.norm3 = norm_layer(q_dim)
        
        self.attn = TopKAttention(q_dim, k_dim, hidden_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)


        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(q_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(q_dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, q_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, vfeats, topk_mask):

        x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(vfeats), topk_mask))
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x

class VISBlock(nn.Module):
    def __init__(self, v_dim, l_dim, hidden_dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.topk1 = TopKBlock(v_dim, v_dim, hidden_dim, num_heads, mlp_ratio=4, drop_path=drop_path)
        self.topk2 = TopKBlock(v_dim, v_dim, hidden_dim, num_heads, mlp_ratio=4, drop_path=drop_path)
        self.topk3 = TopKBlock(v_dim, v_dim, hidden_dim, num_heads, mlp_ratio=4, drop_path=drop_path)
        self.topk4 = TopKBlock(v_dim, v_dim, hidden_dim, num_heads, mlp_ratio=4, drop_path=drop_path)

        self.selfatt1 = QLBlock(768, 768, hidden_dim, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path)
        self.selfatt2 = QLBlock(768, 768, hidden_dim, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path)
        self.selfatt3 = QLBlock(768, 768, hidden_dim, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path)
        self.selfatt4 = QLBlock(768, 768, hidden_dim, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path)

        self.proj_l = nn.Linear(l_dim, 256)
        self.proj_v = nn.Linear(v_dim, 256)

        self.tau = nn.Parameter(torch.ones(1)*0.1, requires_grad=True)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, v_feats, l_feats, l_mask, ratio=0.3):
        with torch.no_grad():
            self.tau.clamp_(0.001, 0.5)
        B, N, C = v_feats.size()
        _, L, _ = l_feats.size()

        l = F.normalize(self.proj_l(l_feats), dim = -1)
        vis = F.normalize(self.proj_v(v_feats), dim = -1)
        sim_mat = torch.matmul(l, vis.transpose(1,2)) # [B, L, N] 

        _, rank_index = torch.topk(sim_mat, k= int(N*ratio), dim=-1 ) # [B, L, K]
        topk_mask = torch.zeros(B, l_feats.size(1), N, device = 'cuda', dtype=torch.long).scatter_(-1, rank_index, 1.0) # [B, L, N]

        mask = topk_mask.unsqueeze(1)# [B, 1, L, N]
        mask = mask * 1e4 - 1e4
        mask2 = topk_mask * 1e4 - 1e4
        sim_mat_ = F.softmax(sim_mat + mask2, dim=-1) # [B, L, N]
        sim_mat_ = sim_mat_.unsqueeze(-1) # [B, L, N, 1]
        vfeats = torch.repeat_interleave(v_feats.unsqueeze(1), L, dim=1) # [B, L, N, dim]
        topk_mask_ = topk_mask.unsqueeze(-1)
        v_queries =  topk_mask_* sim_mat_ * vfeats # [B, L, N, dim]
        v_queries = torch.sum(v_queries, dim=2) / int(N*ratio) # [B, L, dim]

        v_queries = self.topk1(v_queries, v_feats, mask)
        v_queries = self.topk2(v_queries, v_feats, mask)
        v_queries = self.topk3(v_queries, v_feats, mask)
        v_queries = self.topk4(v_queries, v_feats, mask)

        v_queries = self.selfatt1(v_queries, v_queries, l_mask)
        v_queries = self.selfatt2(v_queries, v_queries, l_mask)
        v_queries = self.selfatt3(v_queries, v_queries, l_mask)
        v_queries = self.selfatt4(v_queries, v_queries, l_mask)

        q_mask = torch.cat([l_mask, l_mask], dim=-1)
        queries = torch.cat([l_feats, v_queries], dim=1) # [B, 2L, dim]   

        return queries, q_mask, sim_mat[:,0].squeeze(-1)/ self.tau

class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=768):
        super(Decoder, self).__init__()
        mlp_ratio = 4
        
        self.cross = nn.ModuleList()
        for i in range(2):
            self.cross.append(QLBlock(768, 768, 768, 12, mlp_ratio=mlp_ratio, drop_path=0.1))
            self.cross.append(Block(768, 768, 768, 12, mlp_ratio=mlp_ratio, drop_path=0.1))

        self.VIS = VISBlock(hidden_channels, 768, 768, 12, mlp_ratio=4, drop_path=0.1)
        
        self.cross1 = Block(hidden_channels, 768,  768, 12, mlp_ratio=mlp_ratio, drop_path=0.1)
        self.cross2 = Block(hidden_channels, 768,  768, 12, mlp_ratio=mlp_ratio, drop_path=0.1)
        self.cross3 = Block(hidden_channels, 768,  768, 12, mlp_ratio=mlp_ratio, drop_path=0.1)
        self.cross4 = Block(hidden_channels, 768,  768, 12, mlp_ratio=mlp_ratio, drop_path=0.1)

        self.conv1_4 = nn.Sequential(nn.Conv2d(hidden_channels + in_channels//2, hidden_channels, 3, padding=1, bias=False), 
                                     nn.BatchNorm2d(hidden_channels), nn.ReLU(),
                                     nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(hidden_channels),
                                     nn.ReLU())

        self.conv1_3 = nn.Sequential(nn.Conv2d(hidden_channels + in_channels//4, hidden_channels, 3, padding=1, bias=False), 
                                     nn.BatchNorm2d(hidden_channels), nn.ReLU(),
                                     nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(hidden_channels),
                                     nn.ReLU())
        
        self.conv1_2 = nn.Sequential(nn.Conv2d(hidden_channels + in_channels//8, hidden_channels, 3, padding=1, bias=False), 
                                     nn.BatchNorm2d(hidden_channels), nn.ReLU(),
                                     nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(hidden_channels),
                                     nn.ReLU())
        
        self.conv1_1 =  nn.Conv2d(hidden_channels, 2, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, c4, c3, c2, c1, h, w, l_feats, l_mask, training): 

        B = c4.shape[0]
        h4, w4 = h//8, w//8
        h3, w3 = h//4, w//4
        h2, w2 = h//2, w//2

        mask_l = l_mask.permute(0,2,1) # [B, 1, L]
        mask_l = mask_l.unsqueeze(1)
        mask_l = 1e4*mask_l - 1e4
        
        for i in range(0,3,2):
            l_feats=self.cross[i](l_feats, c4, None)
            c4 = self.cross[i+1](c4, l_feats, mask_l, h4, w4)
     
        new_query, new_mask, sim_mat1 = self.VIS(c4, l_feats, mask_l, ratio=0.3)
        c4_ = self.cross4(c4, new_query, new_mask, h4, w4)
        x = c4_.permute(0, 2, 1).reshape(B, -1, h4, w4)

        x = F.interpolate(input=x, size=(h3, w3), mode='bilinear', align_corners=True)
        x = torch.cat([x, c3], dim=1)
        x = self.conv1_4(x)

        c3_ = x.flatten(2).transpose(1,2)
        c3_ = self.cross3(c3_, new_query, new_mask,  h3, w3)
        x = c3_.permute(0, 2, 1).reshape(B, -1, h3, w3)

        x = F.interpolate(input=x, size=(h2, w2), mode='bilinear', align_corners=True)
        x = torch.cat([x, c2], dim=1)
        x = self.conv1_3(x)

        c2_ = x.flatten(2).transpose(1,2)
        c2_ = self.cross2(c2_, new_query, new_mask, h2, w2)
        x = c2_.permute(0,2,1).reshape(B, -1, h2, w2)

        x = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=True)
        x = torch.cat([x, c1], dim=1)
        x = self.conv1_2(x)

        c1_ = x.flatten(2).transpose(1, 2)
        c1_ = self.cross1(c1_, new_query, new_mask, h, w)
        x = c1_.permute(0,2,1).reshape(B, -1, h, w)

        out = self.conv1_1(x)


        if training:
            sim_mat1 = sim_mat1.reshape(B, 1, h4, w4)
            return out, sim_mat1

        else:
            return out
        
        

