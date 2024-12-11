from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel

from timm.models.layers import  trunc_normal_
from .decoder import Decoder
    
class _METRISDecode(nn.Module):
    def __init__(self, backbone, args):
        super(_METRISDecode, self).__init__()
        self.backbone = backbone
        self.net = Decoder(in_channels=1024)
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

    def forward(self, x, text, l_mask, training=True):
        input_shape = x.shape[-2:]

        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask) 
        l_mask = l_mask.unsqueeze(dim=-1) # (B, L, 1)
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        if training:
            
            x, sim1 = self.net(x_c4, x_c3, x_c2, x_c1, input_shape[0]//4, input_shape[1]//4, l_feats, l_mask, training)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            sim1 = F.interpolate(sim1, size=input_shape, mode='bilinear', align_corners=True)

            return  x, sim1.squeeze(1)
        
        x = self.net(x_c4, x_c3, x_c2, x_c1, input_shape[0]//4, input_shape[1]//4, l_feats, l_mask, training)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x

class METRIS(_METRISDecode):
    pass

