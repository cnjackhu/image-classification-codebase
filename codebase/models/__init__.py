import timm
import torch
import torch.hub
from torchvision.models import (
    mobilenet_v2,
    resnet18,
    resnet50,
    shufflenet_v2_x1_0,
    swin_t,
    vit_b_16,
)

from .dummy_model import dummy_model
from .register import MODEL


@MODEL.register
def PyTorchHub(repo: str, name: str, **kwargs):
    return torch.hub.load(repo, name, **kwargs)


@MODEL.register
def TimmHub(name: str, **kwargs):
    return timm.create_model(name, pretrained=False)

@MODEL.register
def Resnet50(name:str,**kwargs):
    return resnet50(pretrained=False)

MODEL.register(resnet18)
MODEL.register(resnet50)
MODEL.register(mobilenet_v2)
MODEL.register(shufflenet_v2_x1_0)
MODEL.register(vit_b_16)
# MODEL.register(swin_t)
