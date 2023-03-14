import torch
from utils import load_checkpoint, save_some_validation_examples
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator

# from generator_model_transformer import Generator
from generator_model_self_attention import Generator
# from generator_model_attention import Generator

# from generator_model_MSA_SE import Generator
# from discriminator_model import Discriminator
from torch.utils.data import DataLoader

import time

torch.backends.cudnn.benchmark = True


def main():
    # disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    # opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999), )
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    if config.LOAD_MODEL:
        # load_checkpoint(
        #    "checkpoints/weights_MSA_SE/gen_pix2pix_transformer-400.pth.tar", gen, opt_gen, config.LEARNING_RATE,
        # )
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )

    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    save_some_validation_examples(gen, val_loader, folder="predict_cityscape/predict_transcgan_WGAN_cityscape")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"running time{end - start}")



