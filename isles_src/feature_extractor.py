
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.networks.nets import resnet10, resnet18, resnet34
except ImportError as e:
    raise ImportError("`pip install monai`") from e

_ARCH = {"resnet10": resnet10, "resnet18": resnet18, "resnet34": resnet34}


def _strip_prefix(state_dict: dict, prefix: str = "module.") -> dict:
    
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


def load_local_medicalnet_weights(model: nn.Module, weight_path: str, verbose: bool = True):

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f" {weight_path}\n"
            f"config.py pretrained_weights_path "
        )

    ckpt = torch.load(weight_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state_dict = _strip_prefix(state_dict, "module.")

    model_dict = model.state_dict()
    matched, skipped_shape = {}, []
    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                matched[k] = v
            else:
                skipped_shape.append(k)
    missing = [k for k in model_dict if k not in matched]

    model_dict.update(matched)
    model.load_state_dict(model_dict)

    if verbose:
        print(f"[ResNet3DMultiLayerFeature] : {weight_path}")
        print(f"  {len(matched)} / {len(model_dict)}")
        if skipped_shape:
            print(f" shape : {len(skipped_shape)} ")
        if missing:
            print(f" {len(missing)}")

    return list(matched.keys()), missing


class ChannelProjector(nn.Module):

    def __init__(self, in_channels: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(1, int(in_channels * hidden_ratio))
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor, out_dhw) -> torch.Tensor:
        x = self.proj(feat)  # (B, 1, d, h, w)
        x = F.interpolate(x, size=out_dhw, mode="trilinear", align_corners=False)
        return torch.sigmoid(x)  # [0,1] 


class ResNet3DMultiLayerFeature(nn.Module):
    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = True,
        pretrained_path: str = None,
        n_input_channels: int = 1,
        freeze: bool = True,
        proj_hidden_ratio: float = 0.5,
    ):
        super().__init__()
        if arch not in _ARCH:
            raise ValueError(f"arch: {arch}. {list(_ARCH.keys())} ")

        self.net = _ARCH[arch](
            pretrained=False,
            spatial_dims=3,
            n_input_channels=n_input_channels,
            feed_forward=False,
            shortcut_type="A",
            bias_downsample=False,
        )

        if pretrained:
            if not pretrained_path:
                raise ValueError(
                    "pretrained=True  pretrained_path"
                    " MedicalNet .pth config.py pretrained_weights_path"
                )
            load_local_medicalnet_weights(self.net, pretrained_path)

        self._features = {}
        self.net.layer1.register_forward_hook(self._make_hook("layer1"))
        self.net.layer2.register_forward_hook(self._make_hook("layer2"))
        self.net.layer3.register_forward_hook(self._make_hook("layer3"))

        self.freeze = freeze
        if freeze:
            for p in self.net.parameters():
                p.requires_grad = False
            self.net.eval()

        with torch.no_grad():
            was_training = self.net.training
            self.net.eval()
            dummy = torch.zeros(1, n_input_channels, 32, 32, 32)
            self._features = {}
            _ = self.net(dummy)
            ch1 = self._features["layer1"].shape[1]
            ch2 = self._features["layer2"].shape[1]
            ch3 = self._features["layer3"].shape[1]
            self.net.train(was_training)

        self.proj1 = ChannelProjector(ch1, proj_hidden_ratio)
        self.proj2 = ChannelProjector(ch2, proj_hidden_ratio)
        self.proj3 = ChannelProjector(ch3, proj_hidden_ratio)

    def _make_hook(self, name):
        def _hook(module, inp, out):
            self._features[name] = out
        return _hook

    @property
    def projector_parameters(self):

        return (
            list(self.proj1.parameters())
            + list(self.proj2.parameters())
            + list(self.proj3.parameters())
        )

    @property
    def backbone_parameters(self):

        return list(self.net.parameters())

    def _extract(self, x: torch.Tensor):
        self.net.eval() if self.freeze else None
        self._features = {}
        _ = self.net(x)
        return self._features["layer1"], self._features["layer2"], self._features["layer3"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        d, h, w = x.shape[-3:]

        if self.freeze:
            with torch.no_grad():
                f1, f2, f3 = self._extract(x)
        else:
            f1, f2, f3 = self._extract(x)

        c1 = self.proj1(f1, (d, h, w))
        c2 = self.proj2(f2, (d, h, w))
        c3 = self.proj3(f3, (d, h, w))
        return torch.cat([c1, c2, c3], dim=1)  
