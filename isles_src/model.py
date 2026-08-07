
import torch.nn as nn
from monai.networks.nets import UNet as MonaiUNet

from isles_src.feature_extractor import ResNet3DMultiLayerFeature


class ResNetUNet3D3Ch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feature_extractor = ResNet3DMultiLayerFeature(
            arch=cfg.resnet_arch,
            pretrained=cfg.pretrained_resnet,
            pretrained_path=cfg.pretrained_weights_path,
            n_input_channels=cfg.n_input_channels,
            freeze=cfg.freeze_resnet,
            proj_hidden_ratio=cfg.proj_hidden_ratio,
        )
        self.unet = MonaiUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            channels=cfg.unet_channels,
            strides=cfg.unet_strides,
            num_res_units=cfg.unet_num_res_units,
        )
        self.freeze_resnet = cfg.freeze_resnet

    @property
    def trainable_parameters(self):

        params = list(self.unet.parameters())
        params += self.feature_extractor.projector_parameters
        if not self.freeze_resnet:
            params += self.feature_extractor.backbone_parameters
        return params

    def forward(self, x):

        feat3ch = self.feature_extractor(x)  # (B, 3, D, H, W)  
        logits = self.unet(feat3ch)            # (B, 1, D, H, W) 
        return logits
