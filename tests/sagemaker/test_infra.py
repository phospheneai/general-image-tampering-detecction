"""
Static checks for the SageMaker pipeline: the pieces live in different files
(ASL, IAM, configs, workflows, setup script) and only work if they agree.
Each test pins one of those agreements. Needs only pyyaml + pytest; the
metric-format test also runs when torch/peft are installed.

    pytest tests/sagemaker -q
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
INFRA = ROOT / "infra"
SM = ROOT / "sagemaker"
WORKFLOWS = ROOT / ".github" / "workflows"

ACCOUNT, REGION = "123456789012", "us-east-1"

# (state machine name, ASL file, task state, config the ASL runs)
PIPELINES = [
    ("forgery-training", "train-pipeline.asl.json", "Train", "config/normal/train_forensics.yml"),
    ("forgery-smoke", "smoke-pipeline.asl.json", "SmokeTrain", "config/smoke.yml"),
]


def render(path: Path) -> str:
    # the Python twin of infra/render.sh
    return path.read_text(encoding="utf-8").replace("${AWS_ACCOUNT_ID}", ACCOUNT).replace("${AWS_REGION}", REGION)


def asl(name: str) -> dict:
    return json.loads(render(INFRA / name))


def task_params(name: str, state: str) -> dict:
    return asl(name)["States"][state]["Parameters"]


def config(rel: str) -> dict:
    return yaml.safe_load((SM / rel).read_text(encoding="utf-8"))


def workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------
# Templates
# ------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(INFRA.glob("**/*.json")), ids=lambda p: p.name)
def test_templates_render_to_valid_json(path):
    text = render(path)
    assert "${" not in text, f"undefined placeholder in {path.name}"
    json.loads(text)


@pytest.mark.parametrize("_, name, state, __", PIPELINES)
def test_asl_shape(_, name, state, __):
    m = asl(name)
    assert m["StartAt"] == state
    task = m["States"][state]
    assert task["Resource"] == "arn:aws:states:::sagemaker:createTrainingJob.sync"
    for catch in task["Catch"]:
        assert m["States"][catch["Next"]]["Type"] == "Fail"
    assert task_params(name, state)["RoleArn"] == f"arn:aws:iam::{ACCOUNT}:role/forgery-sagemaker-role"


# ------------------------------------------------------------------
# Image <-> ASL <-> config
# ------------------------------------------------------------------

@pytest.mark.parametrize("_, name, state, cfg", PIPELINES)
def test_entrypoint_and_config_exist_in_the_image(_, name, state, cfg):
    algo = task_params(name, state)["AlgorithmSpecification"]
    script = algo["ContainerEntrypoint"][-1]
    assert script.startswith("/opt/ml/code/")
    assert (ROOT / script.removeprefix("/opt/ml/code/")).is_file()   # Dockerfile copies sagemaker/ to /opt/ml/code/sagemaker/
    assert (SM / cfg).is_file()
    if "ContainerArguments" in algo:
        assert algo["ContainerArguments"][:2] == ["--config", cfg]


def _channel_paths(cfg: dict) -> set[str]:
    paths = []
    for split in ("train", "test"):
        root = cfg["datasets"][split]["dataroot"]
        paths += root if isinstance(root, list) else [root]
    paths.append(cfg["structure"]["backbone"]["model_path"])
    return set(paths)


@pytest.mark.parametrize("_, name, state, cfg", PIPELINES)
def test_every_config_path_is_a_mounted_channel(_, name, state, cfg):
    """Channel `x` appears at /opt/ml/input/data/x — rename one, rename the other."""
    channels = {c["ChannelName"] for c in task_params(name, state)["InputDataConfig"]}
    wanted = set()
    for path in _channel_paths(config(cfg)):
        m = re.fullmatch(r"/opt/ml/input/data/([^/]+)/?", path)
        assert m, f"{cfg}: {path} is not a channel mount"
        wanted.add(m.group(1))
    assert wanted == channels


@pytest.mark.parametrize("_, name, state, cfg", PIPELINES)
def test_checkpoints_go_where_sagemaker_syncs(_, name, state, cfg):
    local = task_params(name, state)["CheckpointConfig"]["LocalPath"]
    assert config(cfg)["train_settings"]["save_checkpoint_folder_path"] == local


def test_launcher_inputs_match_the_pipeline():
    """launch.py reads S3 locations from the yml, the pipeline from the ASL."""
    launch = {k: v["uri"] for k, v in config("config/normal/train_forensics.yml")["sagemaker"]["inputs"].items()}
    pipeline = {c["ChannelName"]: c["DataSource"]["S3DataSource"]["S3Uri"]
                for c in task_params("train-pipeline.asl.json", "Train")["InputDataConfig"]}
    assert launch == pipeline


def test_run_training_offers_only_configs_in_the_image():
    options = workflow("run-training.yml")[True]["workflow_dispatch"]["inputs"]["config"]["options"]
    for opt in options:
        assert (SM / opt).is_file(), opt


# ------------------------------------------------------------------
# IAM grants cover what the jobs touch
# ------------------------------------------------------------------

def _s3_resources(policy: dict, action: str) -> list[str]:
    out = []
    for st in policy["Statement"]:
        actions = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
        if action in actions:
            res = st["Resource"] if isinstance(st["Resource"], list) else [st["Resource"]]
            out += [r.removeprefix("arn:aws:s3:::") for r in res]
    return out


def _covered(uri: str, resources: list[str]) -> bool:
    key = uri.removeprefix("s3://")
    return any(key.startswith(r.rstrip("*")) for r in resources)


@pytest.mark.parametrize("_, name, state, __", PIPELINES)
def test_sagemaker_role_can_read_inputs_and_write_outputs(_, name, state, __):
    policy = json.loads(render(INFRA / "iam" / "sagemaker-policy.json"))
    p = task_params(name, state)
    for c in p["InputDataConfig"]:
        assert _covered(c["DataSource"]["S3DataSource"]["S3Uri"], _s3_resources(policy, "s3:GetObject"))
    for fmt in (p["OutputDataConfig"]["S3OutputPath.$"], p["CheckpointConfig"]["S3Uri.$"]):
        uri = re.search(r"'(s3://[^']*)'", fmt).group(1).replace("{}", "job")
        assert _covered(uri, _s3_resources(policy, "s3:PutObject"))


def test_names_agree_across_setup_workflows_and_policies():
    setup = (INFRA / "setup-aws.sh").read_text(encoding="utf-8")
    for role in ("forgery-github-actions-role", "forgery-stepfunctions-role", "forgery-sagemaker-role"):
        assert f"make_role {role}" in setup

    deployed = {e["name"] for e in workflow("deploy-pipeline.yml")["jobs"]["deploy"]["strategy"]["matrix"]["include"]}
    assert deployed == {p[0] for p in PIPELINES}
    for wf, machine in (("smoke-test.yml", "forgery-smoke"), ("run-training.yml", "forgery-training")):
        assert workflow(wf)["env"]["STATE_MACHINE"] == machine
    for wf in ("build-image.yml", "smoke-test.yml", "run-training.yml"):
        assert workflow(wf)["env"]["ECR_REPOSITORY"] == "forgery-train"

    gh = render(INFRA / "iam" / "github-policy.json")
    assert "stateMachine:forgery-*" in gh and "repository/forgery-train" in gh
    assert "role/forgery-stepfunctions-role" in gh
    assert "role/forgery-stepfunctions-role" in (ROOT / ".github/workflows/deploy-pipeline.yml").read_text()


def test_workflows_use_one_region():
    regions = {workflow(f.name)["env"]["AWS_REGION"] for f in WORKFLOWS.glob("*.yml") if "AWS_REGION" in workflow(f.name).get("env", {})}
    assert regions == {REGION}
    assert f"AWS_REGION:-{REGION}" in (INFRA / "setup-aws.sh").read_text(encoding="utf-8")
    assert config("config/normal/train_forensics.yml")["sagemaker"]["region"] == REGION


# ------------------------------------------------------------------
# Metrics regexes <-> trainer log format
# ------------------------------------------------------------------

def test_metric_regexes_match_the_trainer_format():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("streaming")
    from authgenforge.training.trainer import SegmentationTrainer

    class Stub:
        lines: list[str] = []

        def _log(self, msg):
            self.lines.append(msg)

    stub = Stub()
    metrics = dict(loss=0.25, iou=0.5, f1=0.625, precision=0.75, recall=0.5, accuracy=0.875)
    SegmentationTrainer._print_metrics(stub, "Train", 3, metrics)
    SegmentationTrainer._print_metrics(stub, "Val", 3, metrics)
    log = "\n".join(f"2026-10-01 12:00:00 | INFO | forge | {line}" for line in stub.lines)

    expected = {"loss": 0.25, "iou": 0.5, "f1": 0.625, "precision": 0.75, "recall": 0.5, "accuracy": 0.875}
    for m in task_params("train-pipeline.asl.json", "Train")["AlgorithmSpecification"]["MetricDefinitions"]:
        found = re.findall(m["Regex"], log)
        assert len(found) == 1, m
        assert float(found[0]) == expected[m["Name"].split(":")[1]], m


# ------------------------------------------------------------------
# train.py config patching (spot resume / missing init weights)
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_module():
    spec = importlib.util.spec_from_file_location("sm_train", SM / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_cfg(tmp_path, ckpt_root, want_load=False, init=""):
    cfg = {
        "name": "exp",
        "train_settings": {"save_checkpoint_folder_path": str(ckpt_root), "resume_dataloader": False},
        "pretraining_settings": {"want_load": want_load, "checkpoint_path": init},
    }
    path = tmp_path / "cfg.yml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_fresh_run_uses_the_config_unchanged(train_module, tmp_path):
    path = _write_cfg(tmp_path, tmp_path / "ckpt")
    assert train_module._effective_config(path) == path


def test_restored_checkpoint_turns_on_resume(train_module, tmp_path):
    latest = tmp_path / "ckpt" / "exp" / "latest_checkpoint.pth"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b"x")
    eff = train_module._effective_config(_write_cfg(tmp_path, tmp_path / "ckpt"))
    patched = yaml.safe_load(eff.read_text())
    eff.unlink()
    assert patched["train_settings"]["load_checkpoint_file_path"] == str(latest)
    assert patched["train_settings"]["resume_dataloader"] is True


def test_missing_init_weights_disable_want_load(train_module, tmp_path):
    eff = train_module._effective_config(
        _write_cfg(tmp_path, tmp_path / "ckpt", want_load=True, init=str(tmp_path / "nope.pth")))
    patched = yaml.safe_load(eff.read_text())
    eff.unlink()
    assert patched["pretraining_settings"]["want_load"] is False
