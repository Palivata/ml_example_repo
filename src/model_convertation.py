import argparse
import os
import shutil

import onnx
import openvino as ov
import torch
from onnxsim import simplify
from openvino import Core

from src.config import Config
from src.module import AVAILABLE_MODULES

if __name__ == "__main__":
    print("Converting...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--chkpt_path", type=str, help="path to checkpoint file")
    parser.add_argument("--config_file", type=str, help="path to configuration file")
    parser.add_argument("--output_dir", type=str, help="path to output directory")
    args = parser.parse_args()
    config = Config.from_yaml(args.config_file)

    torch_model = AVAILABLE_MODULES[config.experiment].load_from_checkpoint(args.chkpt_path)

    torch_model = torch_model.cpu().eval()
    os.makedirs(args.output_dir, exist_ok=True)
    torch_input = torch.randn(1, 3, config.height, config.width)
    input_names = ["actual_input"]
    output_names = ["output"]

    torch.onnx.export(
        torch_model,
        torch_input,
        "model.onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
    )

    onnx_model = onnx.load("model.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, "model.onnx")

    ov_model = ov.convert_model("model.onnx")
    path_xml = os.path.join(args.output_dir, f"{config.experiment}_model.xml")
    path_bin = os.path.join(args.output_dir, f"{config.experiment}_model.bin")
    ov.runtime.save_model(ov_model, "/model.xml")
    shutil.copy("/model.bin", path_bin)
    shutil.copy("/model.xml", path_xml)

    print("Model saved")

    ov_input = torch_input.cpu().detach().numpy()
    with torch.no_grad():
        output_data_torch = torch_model(torch_input).cpu().detach().numpy()
    ie = Core()
    compiled_model = ie.compile_model(model=ov_model, device_name="CPU")
    ov_output = compiled_model([ov_input])[compiled_model.output(0)]
    os.remove("model.onnx")
    print(f"Torch output, {output_data_torch}")
    print(f"Ov output, {ov_output}")
