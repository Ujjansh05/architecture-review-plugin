"""Fine-tune Laya from the web-only export without requiring a human labeler."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from arch_review.training import run_training
from arch_review.web_training import validate_web_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="The export subdirectory from prepare_web_training.py")
    parser.add_argument("--base", type=Path, required=True, help="Existing Laya Multilingual checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["head", "full"], default="head",
                        help="head updates about 15M parameters and saves a small overlay; full updates the encoder too")
    parser.add_argument("--epochs", type=int, choices=[1, 2], default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_web_export(args.data)
    if args.mode == "head" and args.device != "cpu":
        raise ValueError("The small-head runner currently supports CPU; full mode also accepts CUDA")
    if args.timeout_seconds < 1:
        raise ValueError("Timeout must be positive")
    args.allow_unreviewed, args.freeze_encoder, args.seed, args.hourly_rate_usd = True, args.mode == "head", 17, 0
    # Fail on overlong states before starting either training mode.
    requested_dry_run = args.dry_run
    args.dry_run = True
    preview = run_training(args)
    preview.update({"model": "Laya Multilingual", "approximate_parameters": 322_000_000,
                    "training_mode": args.mode, "labels": "machine_generated_rule_checked"})
    print(json.dumps(preview, indent=2), flush=True)
    if not preview["all_states_fit"]:
        raise ValueError("Shorten the source examples; a model input exceeds the context budget")
    if requested_dry_run:
        return
    args.dry_run = False
    if args.mode == "head":
        script = Path(__file__).with_name("train_head_diagnostic.py")
        subprocess.run([sys.executable, str(script), "--data", str(args.data), "--base", str(args.base),
                        "--output", str(args.output), "--epochs", str(args.epochs),
                        "--timeout-seconds", str(args.timeout_seconds)], check=True)
    else:
        print(json.dumps(run_training(args), indent=2))


if __name__ == "__main__":
    main()
