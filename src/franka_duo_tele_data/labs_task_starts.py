"""Station-owned task-to-dataset mapping for independently verified joint returns."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labs_inference import LabsContract, episode_start, start_identity


@dataclass(frozen=True)
class TaskStart:
    dataset: Path
    state: np.ndarray
    identity: str


def load_task_starts(config, *, task_id=None):
    """Both client and relay read episode 0/frame 0 from the same local catalog."""
    config = Path(config)
    path = config / "task_starts.json"
    if not path.is_file() and task_id is None:
        return {}
    catalog = json.loads(path.read_text())
    if catalog.get("schema") != "labs_task_starts_v1":
        raise ValueError("Unknown task-start catalog schema")
    tasks = catalog["tasks"]
    selected = list(tasks) if task_id is None else [str(task_id)]
    result = {}
    for key in selected:
        if key not in ("1", "2", "3", "4") or key not in tasks:
            raise ValueError(f"No configured first-episode start for task {key}")
        entry = tasks[key]
        dataset = Path(entry["dataset"])
        contract = LabsContract(config, dataset)
        if contract.task != entry["task"]:
            raise ValueError(f"Task {key} instruction disagrees with restore dataset")
        state = episode_start(dataset, 0, contract)
        result[int(key)] = TaskStart(dataset, state, start_identity(state, 0, contract))
    return result
