"""Sinribe entry point.

    sinribe                          launch the GUI
    sinribe AUDIO [-o DIR]           transcribe a local file, headless
    sinribe URL   [-o DIR]           download a podcast/video first, then transcribe it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config, save_config


def _cli(args: argparse.Namespace) -> int:
    from .pipeline import Cancelled, JobSpec, Runner, StageError

    cfg = load_config()
    if args.model:
        cfg["asr_model"] = args.model
    if args.quality:
        cfg["speed_target"] = args.quality
    if args.hotwords:
        cfg["hotwords"] = args.hotwords
    if args.language:
        cfg["language"] = args.language
    if args.speakers:
        cfg["speaker_mode"] = "exact"
        cfg["num_speakers"] = args.speakers
    if args.enrich:
        cfg["llm_enrich"] = True
    if args.no_enrich:
        cfg["llm_enrich"] = False

    from .fetch import is_url

    out_dir = Path(args.output or cfg["output_dir"]).expanduser()
    if is_url(args.audio):
        spec = JobSpec(input_path=None, output_dir=out_dir, cfg=cfg, url=args.audio.strip())
    else:
        src = Path(args.audio).expanduser().resolve()
        if not src.is_file():
            print(f"ERROR: no such file: {src}", file=sys.stderr)
            return 2
        spec = JobSpec(input_path=src, output_dir=out_dir, cfg=cfg)

    state = {"phase": ""}

    def on_progress(frac: float, phase: str, detail: str) -> None:
        if phase != state["phase"]:
            state["phase"] = phase
            print(f"\n>> {phase}", flush=True)
        bar = int(frac * 40)
        sys.stdout.write(f"\r   [{'#' * bar}{'.' * (40 - bar)}] {frac * 100:5.1f}%  {detail}   ")
        sys.stdout.flush()

    def on_log(msg: str) -> None:
        sys.stdout.write(f"\r   {msg}\n")
        sys.stdout.flush()

    runner = Runner(spec, on_progress=on_progress, on_log=on_log)
    try:
        result = runner.run()
    except Cancelled:
        print("\ncancelled")
        return 130
    except (StageError, Exception) as e:  # noqa: BLE001
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print("\n")
    for p in result["written"]:
        print(f"   wrote {p}")
    for name, s in sorted(result["stats"].items(), key=lambda kv: -kv[1]["seconds"]):
        print(f"   {name}: {s['seconds']:.1f}s ({s['share'] * 100:.0f}%)")
    return 0


def _gui() -> int:
    from PySide6.QtWidgets import QApplication

    from .theme import apply_theme
    from .ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Sinribe")
    app.setApplicationDisplayName("Sinribe")
    apply_theme(app)

    cfg = load_config()
    win = MainWindow(cfg)
    win.show()
    rc = app.exec()
    save_config(win.cfg)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(prog="sinribe", description="Offline transcription with "
                                                             "speaker diarization")
    ap.add_argument("--cli", action="store_true", help="run headless instead of the GUI")
    ap.add_argument("audio", nargs="?",
                    help="audio/video file, or a YouTube/podcast URL (implies --cli)")
    ap.add_argument("-o", "--output", help="output directory")
    ap.add_argument("-m", "--model",
                    help="whisper model, or 'auto' to let --quality choose "
                         "(large-v3 | large-v3-german | large-v3-turbo-german | medium.en | small)")
    ap.add_argument("-q", "--quality", type=int, metavar="RTF",
                    help="target realtime factor: 1 is slowest and most accurate, 12 fastest")
    ap.add_argument("--hotwords", help="comma-separated names and jargon to expect")
    ap.add_argument("-l", "--language", help="language code, or 'auto'")
    ap.add_argument("-s", "--speakers", type=int, help="exact number of speakers")
    ap.add_argument("--enrich", action="store_true", help="force LLM chapters + summary on")
    ap.add_argument("--no-enrich", action="store_true", help="force LLM enrichment off")
    args = ap.parse_args()

    if args.audio:
        return _cli(args)
    if args.cli:
        ap.error("--cli requires an audio file")
    return _gui()


if __name__ == "__main__":
    sys.exit(main())
