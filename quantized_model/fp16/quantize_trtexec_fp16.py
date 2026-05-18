#!/usr/bin/env python3
"""Export best_model.pth to ONNX and build a TensorRT engine with trtexec.

Example:
    python3 quantize_with_trtexec.py

INT8 requires a TensorRT calibration cache or an already quantized Q/DQ ONNX:
    python3 quantize_with_trtexec.py --precision int8 --calib-cache calib.cache
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path("/root/data_with_weight_file/checkpoints/best_model.pth")
DEFAULT_OUTPUT_DIR = Path("/root/data_with_weight_file/quantized")
DEFAULT_ONNX = DEFAULT_OUTPUT_DIR / "best_model.onnx"
DEFAULT_ENGINE = DEFAULT_OUTPUT_DIR / "best_model_fp16.engine"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "trtexec_config_fp16.json"


def build_model(num_classes: int) -> nn.Module:
    try:
        model = r2plus1d_18(weights=None)
    except TypeError:
        model = r2plus1d_18(pretrained=False)
    model.fc = nn.Linear(512, num_classes)
    return model


def load_checkpoint(checkpoint_path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"], checkpoint.get("config", {})
    if isinstance(checkpoint, dict):
        return checkpoint, {}

    raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")


def export_onnx(
    checkpoint_path: Path,
    onnx_path: Path,
    batch_size: int,
    num_frames: int,
    num_classes: int,
    opset: int,
    dynamic_batch: bool,
) -> None:
    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError(
            "The Python package 'onnx' is required for ONNX export. "
            "Install it with: python3 -m pip install onnx"
        )

    state_dict, checkpoint_config = load_checkpoint(checkpoint_path)
    num_frames = int(checkpoint_config.get("num_frames", num_frames))
    num_classes = int(checkpoint_config.get("num_classes", num_classes))

    model = build_model(num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, num_frames, 112, 112)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["logits"],
        "dynamic_axes": dynamic_axes,
    }
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path.as_posix(),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(model, dummy_input, onnx_path.as_posix(), **export_kwargs)
    print(f"[OK] ONNX export: {onnx_path}")


def build_trtexec_command(args: argparse.Namespace) -> list[str]:
    engine_path = args.engine
    if args.engine == DEFAULT_ENGINE and args.precision == "int8":
        engine_path = args.engine.with_name("best_model_int8.engine")

    command = [
        args.trtexec,
        f"--onnx={args.onnx}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{args.workspace}",
        "--verbose",
    ]

    if args.dynamic_batch:
        command.extend(
            [
                "--minShapes=input:1x3x16x112x112",
                f"--optShapes=input:{args.batch_size}x3x16x112x112",
                f"--maxShapes=input:{args.max_batch_size}x3x16x112x112",
            ]
        )

    if args.precision == "fp16":
        command.append("--fp16")
    elif args.precision == "int8":
        command.append("--int8")
        if args.calib_cache:
            command.append(f"--calib={args.calib_cache}")

    command.extend(args.extra_trtexec_args)
    return command


def run_trtexec(command: list[str], dry_run: bool) -> None:
    print("[CMD] " + " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def config_path(config: dict, key: str, default: Path | None) -> Path | None:
    value = config.get(key)
    if value is None:
        return default
    return Path(value)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Convert r2plus1d_18 best_model.pth to TensorRT engine via trtexec.",
        parents=[config_parser],
    )
    parser.add_argument("--checkpoint", type=Path, default=config_path(config, "checkpoint", DEFAULT_CHECKPOINT))
    parser.add_argument("--onnx", type=Path, default=config_path(config, "onnx", DEFAULT_ONNX))
    parser.add_argument("--engine", type=Path, default=config_path(config, "engine", DEFAULT_ENGINE))
    parser.add_argument("--trtexec", default=config.get("trtexec", "trtexec"))
    parser.add_argument("--precision", choices=["fp16", "int8"], default=config.get("precision", "fp16"))
    parser.add_argument("--calib-cache", type=Path, default=config_path(config, "calib_cache", None))
    parser.add_argument("--batch-size", type=int, default=config.get("batch_size", 1))
    parser.add_argument("--max-batch-size", type=int, default=config.get("max_batch_size", 8))
    parser.add_argument("--num-frames", type=int, default=config.get("num_frames", 16))
    parser.add_argument("--num-classes", type=int, default=config.get("num_classes", 5))
    parser.add_argument("--opset", type=int, default=config.get("opset", 17))
    parser.add_argument("--workspace", type=int, default=config.get("workspace", 4096), help="Workspace size in MiB.")
    parser.add_argument("--dynamic-batch", action="store_true", default=config.get("dynamic_batch", False))
    parser.add_argument("--static-batch", action="store_false", dest="dynamic_batch")
    parser.add_argument("--skip-onnx-export", action="store_true", default=config.get("skip_onnx_export", False))
    parser.add_argument("--dry-run", action="store_true", default=config.get("dry_run", False))
    parser.add_argument(
        "extra_trtexec_args",
        nargs=argparse.REMAINDER,
        default=config.get("extra_trtexec_args", []),
        help="Extra trtexec flags after '--', for example: -- --profilingVerbosity=detailed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.onnx = args.onnx.resolve()
    args.engine = args.engine.resolve()

    if args.extra_trtexec_args and args.extra_trtexec_args[0] == "--":
        args.extra_trtexec_args = args.extra_trtexec_args[1:]

    if args.precision == "int8" and args.calib_cache is None:
        print(
            "[WARN] INT8 was requested without --calib-cache. "
            "This only works if the ONNX already contains quantization ranges/Q-DQ nodes."
        )

    if not args.skip_onnx_export:
        export_onnx(
            checkpoint_path=args.checkpoint,
            onnx_path=args.onnx,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            num_classes=args.num_classes,
            opset=args.opset,
            dynamic_batch=args.dynamic_batch,
        )

    command = build_trtexec_command(args)
    run_trtexec(command, args.dry_run)
    print("[OK] TensorRT build finished.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] trtexec failed with exit code {exc.returncode}", file=sys.stderr)
        raise
