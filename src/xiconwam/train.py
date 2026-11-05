"""Training entry. `python -m xiconwam.train` reads configs/ next to this project."""

from xiconwam.engine.runtime_tuning import enable_expandable_segments

enable_expandable_segments()

import hydra
from omegaconf import DictConfig

from xiconwam.kit.config_resolvers import register_default_resolvers
from xiconwam.launch import run_training

register_default_resolvers()


@hydra.main(
    config_path="../../configs",
    config_name="task/robodojo_a800",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
