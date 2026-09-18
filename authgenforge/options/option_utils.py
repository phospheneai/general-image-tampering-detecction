from __future__ import annotations

from authgenforge import *


class NoneDict(dict):
    """
    Dictionary that returns None when a missing key is accessed
    as an attribute.
    """

    def __getattr__(self, key):
        return self.get(key, None)


def dict_to_nonedict(data):
    """
    Recursively convert dictionaries and lists into NoneDict-compatible
    structures.
    """

    if isinstance(data, dict):
        return NoneDict(
            {
                key: dict_to_nonedict(value)
                for key, value in data.items()
            }
        )

    if isinstance(data, list):
        return [
            dict_to_nonedict(value)
            for value in data
        ]

    return data


def load_yaml(path):
    """
    Load a YAML configuration file and convert it into a NoneDict.
    """

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        config = {}

    return dict_to_nonedict(config)


def resolve_path(path, base_dir=None):
    """
    Resolve a relative path against base_dir.

    Absolute paths are returned unchanged.
    """

    if path is None:
        return None

    path = Path(path)

    if path.is_absolute():
        return str(path)

    if base_dir is None:
        base_dir = Path.cwd()

    return str((Path(base_dir) / path).resolve())


def dict2str(opt: dict, indent: int = 1) -> str:
    """
    Pretty-print a (possibly nested) config dict for logging.
    """

    pad = "  " * indent
    lines = []

    for key, value in opt.items():

        if isinstance(value, dict):
            lines.append(f"{pad}{key}: [")
            lines.append(dict2str(value, indent + 1).rstrip("\n"))
            lines.append(f"{pad}]")

        else:
            lines.append(f"{pad}{key}: {value}")

    return "\n".join(lines) + "\n"


def parse_yml(opt_path: str):
    """
    Load and normalize a YAML configuration file.

    Relative paths are resolved relative to the directory
    containing the YAML file.
    """

    opt_path = Path(opt_path).resolve()
    base_dir = opt_path.parent

    opt = load_yaml(
        str(opt_path)
    )

    # ----------------------------------------------------------
    # Dataset paths
    # ----------------------------------------------------------

    datasets = opt.get(
        "datasets",
        {},
    )

    for dataset_cfg in datasets.values():

        if not isinstance(
            dataset_cfg,
            dict,
        ):
            continue

        if "dataroot" in dataset_cfg:

            dataroot = dataset_cfg[
                "dataroot"
            ]

            if isinstance(
                dataroot,
                list,
            ):

                dataset_cfg[
                    "dataroot"
                ] = [
                    resolve_path(
                        path,
                        base_dir,
                    )
                    for path in dataroot
                ]

            else:

                dataset_cfg[
                    "dataroot"
                ] = resolve_path(
                    dataroot,
                    base_dir,
                )

    # ----------------------------------------------------------
    # DINOv3 backbone paths
    # ----------------------------------------------------------

    backbone = (
        opt.get(
            "structure",
            {},
        )
        .get(
            "backbone",
            {},
        )
    )

    if "model_path" in backbone:

        backbone["model_path"] = resolve_path(
            backbone["model_path"],
            base_dir,
        )

    # ----------------------------------------------------------
    # Pretraining checkpoint
    # ----------------------------------------------------------

    pretraining = opt.get(
        "pretraining_settings",
        {},
    )

    if (
        "checkpoint_path"
        in pretraining
    ):

        pretraining[
            "checkpoint_path"
        ] = resolve_path(
            pretraining[
                "checkpoint_path"
            ],
            base_dir,
        )

    # ----------------------------------------------------------
    # Training checkpoint paths
    # ----------------------------------------------------------

    train_settings = opt.get(
        "train_settings",
        {},
    )

    for key in (
        "save_checkpoint_folder_path",
        "load_checkpoint_file_path",
    ):

        if key in train_settings:

            train_settings[key] = resolve_path(
                train_settings[key],
                base_dir,
            )

    # ----------------------------------------------------------
    # Evaluation checkpoint / output paths
    # ----------------------------------------------------------

    eval_settings = opt.get(
        "eval_settings",
        {},
    )

    for key in (
        "checkpoint_path",
        "output_dir",
    ):

        if key in eval_settings:

            eval_settings[key] = resolve_path(
                eval_settings[key],
                base_dir,
            )

    # ----------------------------------------------------------
    # Defaults
    # ----------------------------------------------------------

    epoch_settings = opt.setdefault(
        "epoch_settings",
        NoneDict(),
    )

    if epoch_settings.get(
        "resume_state_epoch"
    ) is None:

        epoch_settings[
            "resume_state_epoch"
        ] = 0

    if epoch_settings.get(
        "resume_state_step"
    ) is None:

        epoch_settings[
            "resume_state_step"
        ] = 0

    return opt