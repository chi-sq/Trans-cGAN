import torch
import config
import xlwt
from torchvision.utils import save_image
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import pandas as pd
# img1为真实图像y，img2为合成图像G(x)
def calculate_mae(img1, img2):
    m = np.mean(abs(img1-img2))
    return m
# img1为真实图像y，img2为合成图像G(x)
def calculate_nmse(img1, img2):
    k1 = np.sum((img1-img2)**2)
    k2 = np.sum(img1**2)
    return k1/k2

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization #        [-1,1]->[0,1]
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")   # save_image函数接受的就是归一化为[0,1]的tensor，
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_some_validation_examples(gen, val_loader, folder):
    # x, y = next(iter(val_loader))
    psnr_list = []
    ssmi_list = []
    nmse_list = []
    for i, (x, y) in enumerate(val_loader):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        gen.eval()
        with torch.no_grad():
            y_fake = gen(x)
            psnr = compare_psnr(y.squeeze(0).cpu().numpy(), y_fake.squeeze(0).cpu().numpy())
            ssmi = compare_ssim(y.squeeze(0).cpu().numpy(), y_fake.squeeze(0).cpu().numpy(), channel_axis=0)
            # mae = calculate_mae(y.squeeze(0).cpu().numpy(), y_fake.squeeze(0).cpu().numpy())
            nmse = calculate_nmse(y.squeeze(0).cpu().numpy(), y_fake.squeeze(0).cpu().numpy())
            psnr_list.append(psnr)
            ssmi_list.append(ssmi)
            nmse_list.append(nmse)
            y_fake = y_fake * 0.5 + 0.5  # remove normalization = unnormalization  #
            save_image(y_fake, folder + f"/y_gen_{i}.png")
            save_image(x * 0.5 + 0.5, folder + f"/input_{i}.png")
            save_image(y * 0.5 + 0.5, folder + f"/label_{i}.png")
            # image = torch.cat([x * 0.5 + 0.5, y * 0.5 + 0.5, y_fake], dim=3)
            # save_image(image, folder + f"/input_label_gen{i}.png")
    # ddof=1 -> n 总体  ddof=0  ->(n-1) 样本：无偏估计
    print(f"psnr的均值为{np.mean(psnr_list)},psnr的标准差为{np.std(psnr_list,ddof=1)}")
    print(f"ssmi的均值为{np.mean(ssmi_list)},ssmi的标准差为{np.std(ssmi_list,ddof=1)}")
    print(f"nmse的均值为{np.mean(nmse_list)},nmse的标准差为{np.std(nmse_list, ddof=1)}")

    # data = pd.DataFrame({psnr:psnr_list,ssmi:ssmi_list,nmse:nmse_list})
    # data.to_csv("index_analysis/result_trans-deagan.csv", header=False)

    # print(psnr_list, ssmi_list, nmse_list)
    return psnr_list, ssmi_list, nmse_list




def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    # 不写下面这行，优化器会重用old checkpoint的学习率。
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


'''-------------------------------------------------------------------------'''
'''save_image官方代码'''
# def save_image(
#     tensor: Union[torch.Tensor, List[torch.Tensor]],
#     fp: Union[Text, pathlib.Path, BinaryIO],
#     nrow: int = 8,
#     padding: int = 2,
#     normalize: bool = False,
#     range: Optional[Tuple[int, int]] = None,
#     scale_each: bool = False,
#     pad_value: int = 0,
#     format: Optional[str] = None,
# ) -> None:
#     """Save a given Tensor into an image file.
#
#     Args:
#         tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
#             saves the tensor as a grid of images by calling ``make_grid``.
#         fp (string or file object): A filename or a file object
#         format(Optional):  If omitted, the format to use is determined from the filename extension.
#             If a file object was used instead of a filename, this parameter should always be used.
#         **kwargs: Other arguments are documented in ``make_grid``.
#     """
#     from PIL import Image
#     grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
#                      normalize=normalize, range=range, scale_each=scale_each)
#     # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
#     ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#     im = Image.fromarray(ndarr)
#     im.save(fp, format=format)


'''  type(),.dtype, a.astype("np.float") 三者的用法
type()    返回参数的数据类型
dtype    返回数组中元素的数据类型
astype()    对数据类型进行转换

a
Out[54]:
array([[0.00965715, 0.98557797],
       [0.58014805, 0.7668128 ]])
type(a)
Out[55]: numpy.ndarray
a.dtype
Out[56]: dtype('float64')
a.dtype.type
Out[57]: numpy.float64
'''