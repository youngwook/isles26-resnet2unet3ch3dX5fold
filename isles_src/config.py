from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:

    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    volume_shape: Tuple[int, int, int] = (256, 256, 256) 
    patch_size: Tuple[int, int, int] = (96, 96, 96)  
    intensity_lower_pct: float = 0.1
    intensity_upper_pct: float = 99.9

    resnet_arch: str = "resnet18"
    pretrained_resnet: bool = True   
    pretrained_weights_path: str = "./resnet_18_23dataset.pth"  
    n_input_channels: int = 1
    freeze_resnet: bool = True
    proj_hidden_ratio: float = 0.5
    unet_channels: Tuple[int, ...] = (32, 64, 128, 256, 512)
    unet_strides: Tuple[int, ...] = (2, 2, 2, 2)
    unet_num_res_units: int = 2

    sw_batch_size: int = 1
    sw_overlap: float = 0.5
    dice_threshold: float = 0.5 

cfg = Config()
