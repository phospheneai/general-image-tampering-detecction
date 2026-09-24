import argparse
import sys
from pathlib import Path


# Project root = parent of this notebook's folder. Resolved from __file__
# (not cwd) so this works both as a CLI script run from anywhere and
# inside a Jupyter kernel, whose cwd matches the notebook's own folder.
PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))   # make authgenforge importable in any kernel

# parse_known_args (not parse_args) so this still runs unchanged inside a
# Jupyter kernel, whose own launch args (-f /path/to/connection.json, ...)
# would otherwise be rejected as unrecognized.
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(PROJ_ROOT / "configs" / "normal" / "train_forensics.yml"),
                    help="path to the training yml whose eval_settings block to use")
parser.add_argument("--device", default="cuda")
args, _ = parser.parse_known_args()

CONFIG_PATH = args.config

print(f"Config : {CONFIG_PATH}")
from authgenforge.evals.evaluator import evaluate_from_yml

metrics = evaluate_from_yml(CONFIG_PATH, device=args.device)
print(metrics)
