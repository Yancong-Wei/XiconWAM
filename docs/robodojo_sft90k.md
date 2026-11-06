# Fresh RoboDojo Wan SFT for 90,000 updates

Use `configs/task/robodojo_wan_sft90k.yaml` for a new joint video/action SFT run.
It inherits the existing RoboDojo data, camera layout, 48-D action mapping,
text-cache and Wan/ActionDiT initialization. It does not load an OpenWAM policy
or mix in other datasets. The original `robodojo_wan` recipe remains available
for the earlier experiment.

The new recipe is an initial experimental setting, not a measured H20 throughput
or RoboDojo score guarantee. GPU training still needs a server smoke test.

## Launch

From the repository root with the training environment active and the existing
weights, dataset statistics and text cache prepared:

```bash
# 4 GPUs: micro-batch 4 x 4 ranks x accumulation 8 = global batch 128.
DRY_RUN=1 GPU_IDS=0,1,2,3 bash scripts/train_from_yaml.sh configs/task/robodojo_wan_sft90k.yaml
GPU_IDS=0,1,2,3 bash scripts/train_from_yaml.sh configs/task/robodojo_wan_sft90k.yaml

# 8 GPUs: reduce accumulation to preserve the same global batch.
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_from_yaml.sh configs/task/robodojo_wan_sft90k.yaml \
  gradient_accumulation_steps=4
```

`DRY_RUN` composes the config, checks global batch and generates launch artifacts;
it does not validate media/weights/caches or allocate a training model. Startup
rejects a global batch other than 128 before loading the model. The YAML
`batch_size` is per GPU, and `max_steps` counts optimizer updates, not accumulated
micro-batches. Do not use this new recipe to resume the old 30k run's full state:
the optimizer groups and training schedule have changed. `resume` defaults to
null and the launcher creates a new run directory.

Before the formal run, test the server with a separate short run:

```bash
GPU_IDS=0,1,2,3 bash scripts/train_from_yaml.sh configs/task/robodojo_wan_sft90k.yaml \
  max_steps=20 log_every=1 save_every=20 eval_every=20 eval_num_samples=4
```

That smoke run has its own short LR schedule and must not initialize the formal
run. Confirm that forward/backward, save and offline evaluation complete; then
start the formal command fresh. If micro-batch 4 exceeds VRAM, use
`batch_size=2 gradient_accumulation_steps=16` on four GPUs, still global 128.

## Training and validation

The video tower uses peak LR `1e-5`; the action tower and state encoder use
`1e-4`. Shared module aliases are deduplicated, frozen VAE/text parameters are
excluded, and each trainable parameter belongs to one optimizer group. Warmup
lasts 1,800 optimizer updates; cosine decay ends at 1% of each group's own peak
(`1e-7` video, `1e-6` action). Logs/W&B report each group's current LR separately.
With no group rates requested, existing MoT recipes keep their single optimizer
group. Native PyTorch grouped cosine and DeepSpeed constant warmup now preserve
the requested per-group learning rates.

BF16, ZeRO-2 and gradient checkpointing are enabled. VAE encoding uses
micro-batch 1 to reduce peak VRAM. These settings trade some throughput for
memory; change them after measuring the server, while keeping global batch 128.

Validation uses 64 fixed held-out windows every 3,000 optimizer updates with 10
denoising steps. It is an offline diagnostic, not a task-balanced closed-loop
benchmark. Checkpoints are written every 15,000 updates with three retained;
the final milestones are 60k, 75k and 90k. Increase `checkpoint_save_limit` if all
milestones must remain on disk. Each save includes weights and resumable state.

With the same data population for which 30k updates was about two epochs, the
90k run covers about six epochs and presents 11.52 million windows. Confirm the
actual dataset size, global batch and planned epochs in the startup log. Use
fixed closed-loop tasks to compare checkpoints at 60k, 75k and 90k; lower offline
loss alone does not establish better RoboDojo performance.

Resume an interrupted **new 90k run** using its full state directory and the same
model, optimizer groups, batch settings, dataset and 90k schedule. The full state
is under `runs/robodojo_wan_sft90k/<run_id>/checkpoints/state/step_XXXXXX/`.
Avoid reusing the old 30k scheduler. Record the resolved `config.yaml` and both
LR curves with the evaluation results.
