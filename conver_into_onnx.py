#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "lane" / "lane_detector.pt"
DEFAULT_ONNX_PATH = REPO_ROOT / "models" / "lane" / "lane_detector.onnx"
DEFAULT_OPENVINO_DIR = REPO_ROOT / "models" / "lane" / "lane_detector_openvino"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export lane_detector.pt to ONNX and optionally OpenVINO IR.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the source .pt model.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_ONNX_PATH),
        help="Target ONNX file path.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Square image size used for export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Export batch size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device, for example cpu or 0.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic ONNX input shapes.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the exported ONNX graph when supported.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export FP16 weights when supported by the selected device.",
    )
    parser.add_argument(
        "--export-openvino",
        action="store_true",
        help="Also convert the exported ONNX model to OpenVINO IR (.xml/.bin).",
    )
    parser.add_argument(
        "--openvino-dir",
        default=str(DEFAULT_OPENVINO_DIR),
        help="Directory where the OpenVINO IR files will be written.",
    )
    return parser


def _export_to_onnx(
    model_path: Path,
    output_path: Path,
    imgsz: int,
    opset: int,
    batch: int,
    device: str,
    dynamic: bool,
    simplify: bool,
    half: bool,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'ultralytics'. Install dependencies with 'pip install -r requirements.txt'."
        ) from exc

    model = YOLO(str(model_path))
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            batch=batch,
            device=device,
            dynamic=dynamic,
            simplify=simplify,
            half=half,
        )
    ).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path != output_path:
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(exported_path), str(output_path))
        return output_path

    return exported_path


def _export_to_openvino(onnx_path: Path, output_dir: Path) -> Path:
    try:
        import openvino as ov
        from openvino.frontend import FrontEndManager
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'openvino'. Install dependencies with 'pip install -r requirements.txt'."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    output_xml = output_dir / f"{onnx_path.stem}.xml"

    frontend_manager = FrontEndManager()
    onnx_frontend = frontend_manager.load_by_framework("onnx")
    if onnx_frontend is None:
        raise RuntimeError("OpenVINO ONNX frontend is not available in this environment.")

    input_model = onnx_frontend.load(str(onnx_path))
    ov_model = onnx_frontend.convert(input_model)
    ov.save_model(ov_model, str(output_xml))
    return output_xml


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    openvino_dir = Path(args.openvino_dir).expanduser().resolve()

    if not model_path.is_file():
        parser.error(f"Model not found: {model_path}")

    try:
        onnx_path = _export_to_onnx(
            model_path=model_path,
            output_path=output_path,
            imgsz=args.imgsz,
            opset=args.opset,
            batch=args.batch,
            device=args.device,
            dynamic=args.dynamic,
            simplify=args.simplify,
            half=args.half,
        )
        print(f"ONNX export saved to: {onnx_path}")

        if args.export_openvino:
            openvino_xml = _export_to_openvino(onnx_path=onnx_path, output_dir=openvino_dir)
            print(f"OpenVINO IR saved to: {openvino_xml}")
            print(f"OpenVINO weights saved to: {openvino_xml.with_suffix('.bin')}")
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
