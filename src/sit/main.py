from __future__ import annotations

import argparse
import hashlib
import importlib.util
import queue
import signal
import sys
import threading
from pathlib import Path

from sit.config import load_config


def preflight(config, cameras) -> None:
    config.model.verify_weights()
    for camera in cameras:
        camera.resolve_source()
    missing = [name for name in ("cv2", "numpy", "torch", "ultralytics", "lap")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError("Faltan dependencias de visión: " + ", ".join(missing) + ". Instale .[vision]")


def run_cameras(config, cameras, max_frames: int | None) -> int:
    from sit.capture.rtsp import CaptureStream
    from sit.perception.ultralytics_pipeline import UltralyticsPipeline
    from sit.runner import process_stream
    from sit.storage.csv_events import CsvEvents
    from sit.storage.diagnostics import configure_logging

    logger = configure_logging(config.data_dir, config.timezone)
    sink = CsvEvents(config.data_dir, config.timezone)
    stop = threading.Event()
    failures: queue.Queue[str] = queue.Queue()
    previous_handlers = {}

    def on_signal(_signum, _frame):
        stop.set()

    def worker(camera):
        try:
            pipeline = UltralyticsPipeline(config.model)
            logger.info("Cámara %s iniciada; dispositivo=%s", camera.camera_id, pipeline.device)
            with CaptureStream(camera, stop) as stream:
                count = process_stream(stream, pipeline, sink, camera, config, max_frames=max_frames,
                                       logger=logger)
            logger.info("Cámara %s detenida; frames procesados=%s", camera.camera_id, count)
        except Exception as error:
            failures.put(camera.camera_id)
            stop.set()
            try:
                logger.error("Cámara %s falló (%s): %s", camera.camera_id, type(error).__name__, error)
            except OSError:
                print(f"Cámara {camera.camera_id} falló ({type(error).__name__}); no se pudo escribir el log",
                      file=sys.stderr)

    workers = [threading.Thread(target=worker, args=(camera,), name=f"camera-{camera.camera_id}")
               for camera in cameras]
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, on_signal)
        for thread in workers:
            thread.start()
        while any(thread.is_alive() for thread in workers):
            for thread in workers:
                thread.join(timeout=0.1)
    finally:
        stop.set()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return 1 if not failures.empty() else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SIT: captura y eventos de tráfico")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run", "zones"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("config/cameras.example.json"))
        command.add_argument("--camera", help="ID de cámara; por defecto todas las habilitadas")
        if name == "check":
            command.add_argument("--runtime", action="store_true", help="Comprobar modelo, entorno y dependencias sin conectar")
        if name == "run":
            command.add_argument("--max-frames", type=int)
        if name == "zones":
            command.add_argument("--zone", help="ID de zona; por defecto la primera")
    demo = commands.add_parser("demo", help="Ejemplo determinista de eventos sin cámara ni GPU")
    demo.add_argument("--output", type=Path, default=Path("data/demo"))
    model_info = commands.add_parser("model-info", help="Calcular SHA-256 sin cargar el modelo")
    model_info.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from sit.demo import run_demo
            count, path = run_demo(args.output)
            print(f"Demo completada: {count} eventos en {path}")
            return 0
        if args.command == "model-info":
            with args.path.open("rb") as handle:
                print(hashlib.file_digest(handle, "sha256").hexdigest())
            return 0
        config = load_config(args.config)
        cameras = tuple(c for c in config.cameras if args.camera is None or c.camera_id == args.camera)
        if not cameras:
            raise ValueError("La cámara no existe o está deshabilitada")
        if args.command == "check":
            if args.runtime:
                preflight(config, cameras)
            print("Configuración válida" + (" y recursos locales disponibles (sin conectar)" if args.runtime
                                           else " (estructura; use --runtime para comprobar recursos)"))
            return 0
        if args.command == "zones":
            if len(cameras) != 1:
                raise ValueError("Seleccione una cámara mediante --camera")
            camera = cameras[0]
            camera.resolve_source()
            index = 0
            if args.zone:
                ids = [z.zone_id for z in camera.zones]
                if args.zone not in ids:
                    raise ValueError("La zona no existe")
                index = ids.index(args.zone)
            from sit.ui.zone_editor import edit_zone
            edit_zone(camera, index)
            return 0
        if args.max_frames is not None and args.max_frames <= 0:
            raise ValueError("max-frames debe ser mayor que cero")
        preflight(config, cameras)
        return run_cameras(config, cameras, args.max_frames)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as error:
        # El filtro también protege errores de configuración que accidentalmente incluyan RTSP.
        import re
        text = re.sub(r"rtsps?://[^\s\"']+", "<RTSP omitido>", str(error), flags=re.I)
        parser.exit(2, f"Error: {text}\n")


if __name__ == "__main__":
    raise SystemExit(main())
