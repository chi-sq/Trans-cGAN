import torch
import albumentations as A  ### ?
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/cityscapes/cityscapes/train"
VAL_DIR = "data/cityscapes/cityscapes/val"
LEARNING_RATE = 2e-4
LEARNING_RATE_disc = 2e-4           # 最开始都是2e-4 LEARNING_RATE_disc = 4e-4   LEARNING_RATE_gen = 1e-4
LEARNING_RATE_gen = 2e-4
BATCH_SIZE = 32
NUM_WORKERS = 4
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
EDGE_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
WEIGHT_CLIP = 0.01
LOAD_MODEL = True
SAVE_MODEL = False
CHECKPOINT_DISC = "checkpoints_cityscape/disc_transcgan_WGAN_cityscape.pth.tar"
CHECKPOINT_GEN = "checkpoints_cityscape/gen_transcgan_WGAN_cityscape.pth.tar"
# both_transform = A.Compose(
#     [A.Resize(width=256, height=256), A.HorizontalFlip(p=0.5)], additional_targets={"image0": "image"},
# )
both_transform = A.Compose(
    [
        # A.RandomCrop(width=256, height=256),
        # A.HorizontalFlip(p=0.5),
        A.Resize(width=256, height=256)], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        # 那transform.Normalize()是怎么工作的呢？
        # ToTensor()能够把灰度范围从0-255变换到0-1之间，而后面的transform.Normalize()则把0-1变换到(-1,1).
        # 具体地说，对每个通道而言，Normalize执行以下操作：
        # A.ColorJitter(p=0.2),
        # A.RandomCrop(p=0.2),
        # A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
