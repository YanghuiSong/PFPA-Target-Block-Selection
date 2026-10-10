#!/usr/bin/env python3
"""Frozen 3-class HF Swin-T + UPerNet wrapper for the PFPA supplement."""
from __future__ import annotations
from torch import nn
from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation

DEPTHS=[2,2,6,2]
NUM_HEADS=[3,6,12,24]


def make_upernet():
    backbone=SwinConfig(
        image_size=512, patch_size=4, num_channels=3, embed_dim=96,
        depths=DEPTHS, num_heads=NUM_HEADS, window_size=7,
        out_features=['stage1','stage2','stage3','stage4'],
    )
    cfg=UperNetConfig(
        backbone_config=backbone, num_labels=3,
        id2label={0:'background',1:'crop',2:'weed'},
        label2id={'background':0,'crop':1,'weed':2},
        use_pretrained_backbone=False, auxiliary_in_channels=384,
        hidden_size=512, pool_scales=[1,2,3,6],
    )
    return UperNetForSemanticSegmentation(cfg)


class SwinTUPerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=make_upernet()

    def forward(self,pixel_values):
        return self.net(pixel_values=pixel_values).logits


def get_transformer_blocks(model):
    layers=model.net.backbone.encoder.layers
    blocks=[]
    for si,stage in enumerate(layers,1):
        for bi,block in enumerate(stage.blocks,1):
            params=sum(p.numel() for p in block.parameters())
            blocks.append({
                'block_id':f'S{si}/B{bi}',
                'block_key':f'stage{si}_block{bi}',
                'stage':si,'stage_block':bi,'module':block,
                'parameter_count':int(params),
                'qualified_prefix':f'net.backbone.encoder.layers.{si-1}.blocks.{bi-1}',
            })
    if len(blocks)!=12 or [sum(x['stage']==s for x in blocks) for s in range(1,5)]!=DEPTHS:
        raise RuntimeError('Swin-T UPerNet did not expose all 12 native transformer blocks')
    return blocks
