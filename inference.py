

import glob
import json
from pathlib import Path

import numpy
import torch
import SimpleITK
from monai.inferers import sliding_window_inference
from isles_src.config import Config
from isles_src.model import ResNetUNet3D3Ch
from isles_src.preprocess import (
    build_inference_transforms,
    build_inverter,
    find_input_image_path,
)

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
MODEL_DIR = Path("/opt/ml/model")  
cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold = 0.5
transforms= build_inference_transforms(cfg)

def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


def init_model():
    global threshold
    _show_torch_cuda_info()

    print(f"[inference] using device: {device}")

    ckpt_candidates = sorted(glob.glob(str(MODEL_DIR / "**/*.pt"), recursive=True))
    if not ckpt_candidates:
        raise FileNotFoundError(
            f"{MODEL_DIR} there is no pt file"
        )
    ckpt_path = ckpt_candidates[0]
    print(f"[inference] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    for k, v in ckpt.get("config", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.pretrained_resnet = False

    model = ResNetUNet3D3Ch(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    threshold = float(ckpt.get("best_threshold", cfg.dice_threshold))
    print(
        f"[inference] checkpoint epoch={ckpt.get('epoch')} "
        f"val_fold={ckpt.get('val_fold')} threshold={threshold:.3f}"
    )

    return model


def run(model):
    print(f"[inference] run inference")
    interface_key = get_interface_key()
    handler = {
        ("stroke-metadata", "t1-brain-mri"): interf0_handler,
    }[interface_key]
    return handler(model)


def interf0_handler(model_bundle):
    model = model_bundle

    path_image = str(INPUT_PATH / "images/t1-brain-mri")
    print(f'get image from {path_image}')
    t1_path = find_input_image_path(INPUT_PATH / "images/t1-brain-mri")
    t1_image_ref = SimpleITK.ReadImage(t1_path)

    path_image = str(INPUT_PATH / "stroke-metadata.json")
    print(f'get metadata from {path_image}')
    input_stroke_metadata = load_json_file(location=INPUT_PATH / "stroke-metadata.json")
    print("=+=" * 10)
    print("Loaded Stroke Metadata:")
    print(json.dumps(input_stroke_metadata, indent=2))
    print("=+=" * 10)
    data = transforms({"image": t1_path})
    image_tensor = data["image"].unsqueeze(0).to(device)  
    
    print(f'transform image and inference it')    
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image_tensor,
            roi_size=cfg.patch_size,
            sw_batch_size=cfg.sw_batch_size,
            predictor=model,
            overlap=cfg.sw_overlap,
            mode="gaussian",
        )
        prob = torch.sigmoid(logits)[0]  
    data["pred"] = prob.cpu()
    inverter = build_inverter(transforms)
    data = inverter(data)
    prob_original = data["pred"][0].numpy().astype(numpy.float32) 

    binary_segmentation_mask = (prob_original >= threshold).astype(numpy.uint8)
    print(f'save results')
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/stroke-lesion-segmentation",
        array=binary_segmentation_mask,
        reference_image=t1_image_ref,
    )
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/lesion-probability-map",
        array=prob_original,
        reference_image=t1_image_ref,
    )

    return 0


def get_interface_key():
    path_json = str(INPUT_PATH / "inputs.json")
    print(f'get json from {path_json}')
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["socket"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    with open(location) as f:
        return json.loads(f.read())


def write_array_as_image_file(*, location, array, reference_image=None):
    location.mkdir(parents=True, exist_ok=True)
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)

    if reference_image is not None:
        image.SetSpacing(reference_image.GetSpacing())
        image.SetOrigin(reference_image.GetOrigin())
        image.SetDirection(reference_image.GetDirection())

    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


if __name__ == "__main__":
    raise SystemExit(run(model=init_model()))
