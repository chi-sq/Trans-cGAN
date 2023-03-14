'''baseline:
pix2pix
pix2pix+cattention  baseline + global
pix2pix+AG
pix2pix+self_attention   baseline + global
trans-cGAN
implementation
改进的算法流程相比于原始的GAN的算法实现只改了四点：
1.判别器最后一层去掉sigmoid
2.生成器和判别器的loss不取log
3.每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数C
4.不要用基于动量的优化算法（包括momentum和adam），推荐RMSProp,SGD也行

不用spectral norm而使用weight clip  label smoothing=0.1
'''

import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples, save_some_validation_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
# from generator_model_hybrid import Generator
# from generator_model import Generator
# from generator_model_transformer import Generator
from generator_model_self_attention import Generator
# from generator_model_attention import Generator
from discriminator_model import Discriminator
# from discriminator_model_update import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True
# torch.set_default_tensor_type(torch.FloatTensor)

def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, label_smoothing=0.,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            loss_critic = -(torch.mean(D_real) - torch.mean(D_fake))
            D_loss = loss_critic


        disc.zero_grad()

        # D_loss.backward()
        # opt_disc.step()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        # wgan 权重clamp到一个常数C
        for p in disc.parameters():
            p.data.clamp_(-config.WEIGHT_CLIP, config.WEIGHT_CLIP)

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = -torch.mean(D_fake)
            L1 = l1_loss(y_fake, y)
            G_loss = G_fake_loss + L1 * 0.5

        opt_gen.zero_grad()

        # G_loss.backward()
        # opt_gen.step()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.RMSprop(disc.parameters(), lr=config.LEARNING_RATE_disc)
    opt_gen = optim.RMSprop(gen.parameters(), lr=config.LEARNING_RATE_gen)
    BCE = nn.BCELoss()
    L1_LOSS = nn.L1Loss()

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
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, label_smoothing=0.1
        )
        print(f"epoch{epoch + 1} done!!!")

        if config.SAVE_MODEL and epoch % 5 == 0:
            # save_checkpoint(gen, opt_gen, filename="weights_trans/gen_pix2pix_transformer-{}.pth.tar".format(epoch))
            # save_checkpoint(disc, opt_disc, filename="weights_trans/disc_pix2pix_transformer-{}.pth.tar".format(epoch))
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
        save_some_examples(gen, val_loader, epoch, folder="evaluation_transcgan_WGAN_cityscape")




if __name__ == "__main__":
    main()