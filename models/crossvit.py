# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Modifed from Timm. https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
import numpy as np
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp, LayerScale, Attention
from . import utils


_model_urls = {
    'crossvit_15_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_224.pth',
    'crossvit_15_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_dagger_224.pth',
    'crossvit_15_dagger_384': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_dagger_384.pth',
    'crossvit_18_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_224.pth',
    'crossvit_18_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_dagger_224.pth',
    'crossvit_18_dagger_384': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_dagger_384.pth',
    'crossvit_9_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_9_224.pth',
    'crossvit_9_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_9_dagger_224.pth',
    'crossvit_base_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_base_224.pth',
    'crossvit_small_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_small_224.pth',
    'crossvit_tiny_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_tiny_224.pth',
}

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class UnPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=14, patch_size=16, in_chans=768, embed_dim=3, multi_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # if multi_conv:
        #     if patch_size[0] == 12:
        #         self.proj = nn.Sequential(
        #             nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
        #         )
        #     elif patch_size[0] == 16:
        #         self.proj = nn.Sequential(
        #             nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
        #         )
        # else:
        self.proj = nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, N_, E = x.shape
        N = int(np.sqrt(N_))

        # FIXME look at relaxing size constraints
        assert N == self.img_size[0], \
            f"Input image size ({N}) doesn't match model ({self.img_size[0]})."
        x = x.transpose(2,1).unflatten(2, (N, N))
        x = self.proj(x) # (B, C, H, W)
        return x
    
#################################################################################
#                       Ordinary Transformer Blocks + AdaLN-0                   #
#################################################################################

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
   
class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            # qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            # norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, t):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.drop_path1(self.ls1(self.attn(modulate(self.norm1(x), shift_msa, scale_msa ))))
        x = x + gate_mlp.unsqueeze(1) * self.drop_path2(self.ls2(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp ))))
        return x


#################################################################################
#                       Cross-Transformer Blocks                                #
#################################################################################

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        x, t = inputs
        for module in self._modules.values():
            x = module(x, t)
        return x

class MultiScaleBlock(nn.Module):

    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, 
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                self.blocks.append(mySequential(*tmp))
                # self.blocks.append(nn.ModuleList(tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d+1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[d]), act_layer(), nn.Linear(dim[d], dim[(d+1) % num_branches])]
            self.projs.append(nn.Sequential(*tmp))

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d+1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                       has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                   has_mlp=False))
                self.fusion.append(nn.Sequential(*tmp))

        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d+1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[(d+1) % num_branches]), act_layer(), nn.Linear(dim[(d+1) % num_branches], dim[d])]
            self.revert_projs.append(nn.Sequential(*tmp))

    def forward(self, x, t):
        outs_b = [block(x_, t_) for x_, t_, block in zip(x, t, self.blocks)]
        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
        return outs


def _compute_num_patches(img_size, patches):
    return [i // p * i // p for i, p in zip(img_size,patches)]


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_chans=3, num_classes=10, embed_dim=(192, 384), depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, multi_conv=False):
        super().__init__()

        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = to_2tuple(img_size)
        self.img_size = img_size

        num_patches = _compute_num_patches(img_size, patch_size)
        self.num_branches = len(patch_size)
        
        self.t_embedder = nn.ModuleList()
        for dim in embed_dim:
            self.t_embedder.append(TimestepEmbedder(dim))

        self.patch_embed = nn.ModuleList()
        if hybrid_backbone is None:
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
            for im_s, p, d in zip(img_size, patch_size, embed_dim):
                self.patch_embed.append(PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d, multi_conv=multi_conv))
        else:
            self.pos_embed = nn.ParameterList()
            from .t2t import T2T, get_sinusoid_encoding
            tokens_type = 'transformer' if hybrid_backbone == 't2t' else 'performer'
            for idx, (im_s, p, d) in enumerate(zip(img_size, patch_size, embed_dim)):
                self.patch_embed.append(T2T(im_s, tokens_type=tokens_type, patch_size=p, embed_dim=d))
                self.pos_embed.append(nn.Parameter(data=get_sinusoid_encoding(n_position=1 + num_patches[idx], d_hid=embed_dim[idx]), requires_grad=False))

            del self.pos_embed
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])

        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim[i])) for i in range(self.num_branches)])
        self.pos_drop = nn.Dropout(p=drop_rate)

        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0

        self.blocks = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_,
                                  norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
        self.head = nn.ModuleList([nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity() for i in range(self.num_branches)])

        self.unpatch_embed = nn.ModuleList()
        if hybrid_backbone is None:
            for im_s, p, d in zip(img_size, patch_size, embed_dim):
                self.unpatch_embed.append(UnPatchEmbed(img_size=im_s//p, patch_size=p, in_chans=d, embed_dim=in_chans, multi_conv=multi_conv))

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)
            trunc_normal_(self.cls_token[i], std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, t):
        B, C, H, W = x.shape
        xs = []
        ts = []
        for i in range(self.num_branches):
            x_ = torch.nn.functional.interpolate(x, size=(self.img_size[i], self.img_size[i]), mode='bicubic') if H != self.img_size[i] else x
            tmp = self.patch_embed[i](x_)
            cls_tokens = self.cls_token[i].expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            
            t_ = self.t_embedder[i](t)
            
            # tmp += t_
            
            xs.append(tmp)
            ts.append(t_)

        for blk in self.blocks:
            xs = blk(xs, ts)

        # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
        xs = [self.norm[i](x) for i, x in enumerate(xs)] # x: tokens for each branch (# of branches = 2, len(xs) = 2)
        # out = [x[:, 0] for x in xs] # CLS tokens for each branch

        # xs = (2, B, 1 + N^2, E)
        xs = [x[:,1:,:] for x in xs] # (2, B, N^2, E)
        for i in range(self.num_branches):
            xs[i] = self.unpatch_embed[i](xs[i])
        # (2, B, C, H, W)
        out = xs

        return out

    def forward(self, x, t):
        xs = self.forward_features(x, t)
        # ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
        # ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)
        
        score_sp = xs[0]
        score_freq = xs[1]

        return score_sp


@utils.register_model(name='crossvit')
def crossvit_32(config, **kwargs):
    model = VisionTransformer(img_size=[32, 32],
                              patch_size=[2, 4], embed_dim=[96, 192], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[3, 3], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    # if pretrained:
    #     state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_tiny_224'], map_location='cpu')
    #     model.load_state_dict(state_dict)
    return model

# @register_model
def crossvit_tiny_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[96, 192], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[3, 3], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_tiny_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


# @register_model
def crossvit_small_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[6, 6], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_small_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


# @register_model
def crossvit_base_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[384, 768], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[12, 12], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_base_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


# @register_model
def crossvit_9_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[128, 256], depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
                              num_heads=[4, 4], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_9_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


# @register_model
def crossvit_15_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_15_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


# @register_model
def crossvit_18_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_18_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


# @register_model
def crossvit_9_dagger_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[128, 256], depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
                              num_heads=[4, 4], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_9_dagger_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model

# @register_model
def crossvit_15_dagger_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_15_dagger_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model

# @register_model
def crossvit_15_dagger_384(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[408, 384],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_15_dagger_384'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model

# @register_model
def crossvit_18_dagger_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_18_dagger_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model

# @register_model
def crossvit_18_dagger_384(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[408, 384],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_18_dagger_384'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model
