'''deagan'''

import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples, save_some_validation_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset

from generator_model_MSA_SE import Generator, sobelLayer
from discriminator_model_update import Discriminator

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type(torch.FloatTensor)

def train_fn(
    epoch, disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, sober_loss, g_scaler, d_scaler,
    label_smoothing=0.,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        edge_y = sobelLayer(y)
        # Train Discriminator
        # with torch.cuda.amp.autocast():  #  半精度加速训练
        y_fake = gen(x)
        edge_y_fake = sobelLayer(y_fake)
        D_real = disc(x, y, edge_y)
        D_real_loss = bce(D_real, torch.ones_like(D_real) - label_smoothing * torch.rand_like(D_real))
        D_fake = disc(x, y_fake.detach(), edge_y_fake.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake) + label_smoothing * torch.rand_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2
        # devide by 2 make the discriminator train slower contrast to relative generator

        disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # Train generator
        # with torch.cuda.amp.autocast():
        D_fake = disc(x, y_fake,edge_y_fake)
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
        L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
        Sober_loss = sober_loss(edge_y, edge_y_fake) * config.EDGE_LAMBDA * min(epoch*0.005, 1)
        G_loss = G_fake_loss + L1 + Sober_loss

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    sober_Loss = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            epoch, disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, sober_Loss, g_scaler, d_scaler,
            label_smoothing=0.1
        )

        print(f"epoch{epoch + 1} done!!!")
        if config.SAVE_MODEL and (epoch+1) % 100 == 0:
            save_checkpoint(gen, opt_gen, filename="weights_MSA_SE/gen_pix2pix_transformer-{}.pth.tar".format(epoch+1))
            save_checkpoint(disc, opt_disc, filename="weights_MSA_SE/disc_pix2pix_transformer-{}.pth.tar".format(epoch+1))


        save_some_examples(gen, val_loader, epoch, folder="evaluation_trans-cgan+edge")

if __name__ == "__main__":
    main()
