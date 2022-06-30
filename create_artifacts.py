import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/data_preparation", config_name="config.yaml")
def main(config: DictConfig):

    from src.data_preparation import prepare_data
    from src.utils import utils


    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    return prepare_data(config)


if __name__ == "__main__":
    main()
