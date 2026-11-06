# Nested LeRobot datasets

`dataset_dirs` accepts either LeRobot v3 dataset roots or containers at arbitrary
depth. Training, normalization-stat preparation, and text precomputation share
the same resolver. A root is recognized by `meta/info.json` with
`codebase_version: v3.0`; directory names and depth do not determine its identity.

Inspect the server's tree without loading models, parquet or video:

```bash
PYTHONPATH=src python -m xiconwam.data.discovery /data/datasets > dataset_inventory.json
```

Replace `/data/datasets` with the real server path. The JSON includes canonical
paths, robot type, FPS, episode/frame counts and feature metadata. This is an
inventory, not a guarantee that media is intact or actions are compatible.
Keep it with the experiment and use its explicit leaf paths when the input tree
may change during training or between a run and its resume.

The resolver handles a direct `Droid_v30/meta/info.json`, nested
`InternData_v30/articulation_tasks/franka/close_the_laptop/meta/info.json`, and
RoboCOIN leaves at any depth. Ordinary files are ignored. Hidden descendants are
excluded (an explicitly supplied hidden root is allowed). Child directories
are sorted; explicit input order is retained. Symlink aliases and overlapping
roots are deduplicated by physical directory identity, and cycles are pruned.
Traversal stops at each dataset root. Missing or empty explicit inputs, malformed
metadata, unsupported versions, and directory read errors abort discovery.

For a homogeneous group, point the existing configuration at its container:

```bash
GPU_IDS=0,1,2,3 bash scripts/train_from_yaml.sh configs/task/robodojo_wan.yaml \
  'data.train.dataset_dirs=[/data/compatible_robodojo_exports]' \
  data.train.pretrained_norm_stats=null
```

This command still requires prepared model weights and text embeddings. The
RoboDojo camera/action configuration must actually match every selected leaf.
Use the same `dataset_dirs` override with `scripts/precompute_text_embeds.sh` or
`scripts/compute_dataset_stats.py`; specify `--output` explicitly when computing
aggregate statistics. Do not reuse a single-dataset statistics file for a new
population without verifying that choice.

Discovery does **not** make DROID, InternData and RoboCOIN one homogeneous
source. Configure each compatible robot/action/camera group separately under
`data=mixture`. One source has one processor and normalization convention;
its leaves must share FPS and compatible feature semantics. The current mixture
template only registers RoboDojo. Dataset-specific adapters and mixture-wide
text-cache preparation remain separate work. In particular, identical vector
width does not imply identical joint/EEF, coordinate-frame, absolute/delta,
gripper, or physical-time semantics.
