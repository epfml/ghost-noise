""" ConvMixer

"""
import torch
import torch.nn as nn

from shared.modules.factories import get_conv_factory, get_linear_factory, get_norm_factory

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d
from ._registry import register_model
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq

__all__ = ['ConvMixer']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        'first_conv': 'stem.0',
        **kwargs
    }


default_cfgs = {
    'convmixer_1536_20': _cfg(url='https://github.com/tmp-iclr/convmixer/releases/download/timm-v1.0/convmixer_1536_20_ks9_p7.pth.tar'),
    'convmixer_768_32': _cfg(url='https://github.com/tmp-iclr/convmixer/releases/download/timm-v1.0/convmixer_768_32_ks7_p7_relu.pth.tar'),
    'convmixer_1024_20_ks9_p14': _cfg(url='https://github.com/tmp-iclr/convmixer/releases/download/timm-v1.0/convmixer_1024_20_ks9_p14.pth.tar')
}


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(
        self, dim, depth, kernel_size=9, patch_size=7, in_chans=3, num_classes=1000, global_pool='avg',
        act_layer=nn.GELU,
        conv_cfg='standard',
        linear_cfg='standard',
        norm_cfg='bn2d',
        **kwargs
    ):
        super().__init__()
        conv_layer = get_conv_factory(conv_cfg)
        linear_layer = get_linear_factory(linear_cfg)
        norm_layer = get_norm_factory(norm_cfg)
        self.num_classes = num_classes
        self.num_features = dim
        self.grad_checkpointing = False

        self.stem = nn.Sequential(
            conv_layer(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            act_layer(),
            norm_layer(num_features=dim),
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        conv_layer(dim, dim, kernel_size, groups=dim, padding="same"),
                        act_layer(),
                        norm_layer(num_features=dim),
                    )),
                    conv_layer(dim, dim, kernel_size=1),
                    act_layer(),
                    norm_layer(num_features=dim),
            ) for _ in range(depth)]
        )
        self.pooling = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        self.head = linear_layer(dim, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^stem', blocks=r'^blocks\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.pooling = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
          
    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.pooling(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_convmixer(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ConvMixer, variant, pretrained, **kwargs)


@register_model
def convmixer_1536_20(pretrained=False, **kwargs):
    model_args = dict(dim=1536, depth=20, kernel_size=9, patch_size=7, **kwargs)
    return _create_convmixer('convmixer_1536_20', pretrained, **model_args)


@register_model
def convmixer_768_32(pretrained=False, **kwargs):
    model_args = dict(dim=768, depth=32, kernel_size=7, patch_size=7, act_layer=nn.ReLU, **kwargs)
    return _create_convmixer('convmixer_768_32', pretrained, **model_args)


@register_model
def convmixer_1024_20_ks9_p14(pretrained=False, **kwargs):
    model_args = dict(dim=1024, depth=20, kernel_size=9, patch_size=14, **kwargs)
    return _create_convmixer('convmixer_1024_20_ks9_p14', pretrained, **model_args)


@register_model
def convmixer_cifar(pretrained=False, **kwargs):
    assert not pretrained
    # Remove extra TIMM arguments that we don't support
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('drop_rate', None)
    model_args = {'dim':256, 'depth':8, 'kernel_size':9, 'patch_size':1, **kwargs}
    return ConvMixer(**model_args)
