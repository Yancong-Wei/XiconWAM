# Deployment

Minimal XiconWAM bind set (aligned with pi05 serve / client / protocol / fake client):

| File | Role |
|------|------|
| `model_server.py` | XiconWAM ZMQ policy server (default serial open-loop) |
| `robot_client.py` | Real robot client (`robot_control` required) |
| `protocol.py` | Shared pickle-ZMQ contract (v4) |
| `fake_lerobot_client.py` | Offline open-loop client (dataset → model_server) |

Default control mode is **serial open-loop**: `--execute-steps` equals
`--action-horizon` (K=H). Episode start uses `chunk_index=0` with
`execute_steps=0`. Server `action_fps` must equal client `--execute-fps`.

## Serve

```bash
# Protocol-only (no checkpoint)
python deployment/model_server.py --dry-zeros --action-fps 25 --no-warmup

# Real checkpoint
python deployment/model_server.py \
  --run-dir runs/robodojo/<run_id> \
  --bind tcp://*:5556 \
  --action-horizon 32 --action-dim 14 --action-fps 25 \
  --warmup
```

## Offline fake client

```bash
python deployment/fake_lerobot_client.py \
  --url tcp://127.0.0.1:5556 \
  --lerobot-dataset-path playground/datasets/robodojo \
  --episode-id 0 --frame-id 0 --chunk-num 3 \
  --execute-steps 32 \
  --instruction "pick up the cube and place it on the platform" \
  --output-dir output/openloop_eval/xiconwam
```

## Robot client

Needs `robot_control` + `configs/robots` (set `ROBOT_CONTROL_ROOT` if needed):

```bash
python deployment/robot_client.py \
  --policy-url tcp://<gpu-host>:5556 \
  --task "pick up the cube and place it on the platform" \
  --action-horizon 32 --execute-steps 32 --execute-fps 25 \
  --chunk-num 0 --dry-run
```
