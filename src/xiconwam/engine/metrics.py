"""Action errors in physical units, with time and unify-slot padding excluded."""

import torch


def action_error_metrics(
    processor, prediction, target, proprio, time_pad=None, dim_pad=None
):
    prediction, target, proprio = [
        value.detach().to(device="cpu", dtype=torch.float32)
        for value in (prediction, target, proprio)
    ]
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("Predicted/target actions must have the same [T,D] shape")
    steps, width = target.shape
    if proprio.ndim != 3 or proprio.shape[0] != 1:
        raise ValueError("Evaluation proprio must be [1,T,D]")
    valid_time = torch.ones(steps, dtype=torch.bool)
    if time_pad is not None:
        if tuple(time_pad.shape) != (steps,):
            raise ValueError("Action time padding must have shape [T]")
        valid_time = ~time_pad.detach().cpu().bool()
    valid_dim = torch.ones(width, dtype=torch.bool)
    if dim_pad is not None:
        if tuple(dim_pad.shape) != (width,):
            raise ValueError("Action dimension padding must have shape [D]")
        valid_dim = ~dim_pad.detach().cpu().bool()

    def unpack(action):
        return processor.action_state_merger.backward(
            {"action": action[None], "state": proprio.clone()}
        )

    # Decode into declared fields without padding back to the unify head.
    pred = processor.normalizer.backward(unpack(prediction))["action"]
    truth = processor.normalizer.backward(unpack(target))["action"]
    valid_fields = unpack(valid_dim.float()[None])["action"]
    metrics, weights = {}, {}
    abs_sum = sq_sum = count = 0
    for meta in processor.shape_meta["action"]:
        key = meta["key"]
        difference = pred[key][0] - truth[key][0]
        valid = valid_time[:, None] & valid_fields[key][0].bool()
        values = difference[valid]
        if not torch.isfinite(values).all():
            raise ValueError(
                f"Non-finite action prediction/target in valid field {key!r}"
            )
        abs_sum += float(values.abs().sum())
        sq_sum += float(values.square().sum())
        count += values.numel()
        for axis in range(difference.shape[-1]):
            axis_values = difference[:, axis][valid[:, axis]]
            if axis_values.numel():
                for label, value in (
                    ("l1", axis_values.abs().mean()),
                    ("l2", axis_values.square().mean()),
                ):
                    name = f"action/{key}/{axis}/{label}"
                    metrics[name], weights[name] = float(value), axis_values.numel()
    if count:
        metrics.update(action_l1=abs_sum / count, action_l2=sq_sum / count)
        weights.update(action_l1=count, action_l2=count)
    return metrics, weights
