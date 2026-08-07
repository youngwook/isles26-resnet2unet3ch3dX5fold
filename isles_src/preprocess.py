import glob

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    ResizeWithPadOrCropd,
    Invertd,
)

IMAGE_KEY = "image"


def build_inference_transforms(cfg) -> Compose:
    
    return Compose(
        [
            LoadImaged(keys=[IMAGE_KEY], image_only=False, ensure_channel_first=False),
            EnsureChannelFirstd(keys=[IMAGE_KEY]),
            Orientationd(keys=[IMAGE_KEY], axcodes="RAS"),
            Spacingd(keys=[IMAGE_KEY], pixdim=cfg.spacing, mode="bilinear"),
            ScaleIntensityRangePercentilesd(
                keys=[IMAGE_KEY],
                lower=cfg.intensity_lower_pct,
                upper=cfg.intensity_upper_pct,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            NormalizeIntensityd(keys=[IMAGE_KEY], nonzero=True, channel_wise=True),
            ResizeWithPadOrCropd(keys=[IMAGE_KEY], spatial_size=cfg.volume_shape),
        ]
    )


def build_inverter(transforms: Compose) -> Invertd:
    
    return Invertd(
        keys="pred",
        transform=transforms,
        orig_keys=IMAGE_KEY,
        nearest_interp=False,  
    )


def find_input_image_path(location) -> str:

    candidates = (
        glob.glob(str(location / "*.mha"))
        + glob.glob(str(location / "*.nii.gz"))
        + glob.glob(str(location / "*.nii"))
    )
    if not candidates:
        raise FileNotFoundError(f" No valid image file found in : {location}")
    return candidates[0]
