from .model.transformer import Transformer
from config import get_config


if __name__ == "__main__":
    config = get_config()
    model = Transformer(config)
