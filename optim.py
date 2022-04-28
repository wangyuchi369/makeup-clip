from invimg.scripts.inference import invert
import argparse

import math
import os

import torch
import torchvision
from torch import optim
from tqdm import tqdm
import numpy as np
from optimclip.criteria.clip_loss import CLIPLoss
from optimclip.criteria.id_loss import IDLoss
from optimclip.models.stylegan2.model import Generator
import clip



class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='控制参数')

    def get_args(self):
        """
        初始化参数
        Returns: parser
        """
        self.parser.add_argument('--input_img', type=str, required=True, help='输入图片的路径和文件名')
        self.parser.add_argument("--text", type=str, required=True, nargs='+', help="描述修改的文本")
        self.parser.add_argument("--gan_model", type=str, default="optimclip/pretrained_models/stylegan2-ffhq-config-f.pt",
                                 help="预训练的stylegan模型")
        self.parser.add_argument("--size", type=int, default=1024, help="图片分辨率")
        self.parser.add_argument("--alpha", type=float, default=0.1, help='初始学习率')
        self.parser.add_argument("--step", type=int, default=30, help="迭代次数")
        self.parser.add_argument("--latent_lambda", type=float, default=0.008,
                                 help="latent-code损失的系数")
        self.parser.add_argument("--img_lambda", type=float, default=0, help="图片损失的系数")
        self.parser.add_argument("--id_lambda", type=float, default=0.001, help="面部损失的系数")
        self.parser.add_argument("--results", type=str, default='results', help="结果放置的文件夹")
        self.parser.add_argument('--id_model', default='optimclip/pretrained_models/model_ir_se50.pth', type=str,
                                 help="图像识别网络")
        self.parser.add_argument("--save_intermediate_image_every", type=int, default=20,
                            help="每隔一定步数保存结果")
        return self.parser.parse_args()




invert()
def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def get_ganmodel(opts):
    generator = Generator(opts.size, 512, 8, channel_multiplier=2)
    # TODO 看看generator
    model = torch.load(opts.gan_model)['g_ema']
    generator.load_state_dict(model, strict=True)
    generator = generator.eval().cuda()
    return generator

if __name__ == '__main__':
    opts = Options().get_args()
    # 分词并拼接
    edit_text = torch.cat([clip.tokenize(opts.text)]).cuda()

    orig_pic = str(opts.input_img).split('/')[-1]
    latent_path = 'invimg/results/latents.npy'
    latents = np.load(latent_path, allow_pickle=True).item()
    latent_code = np.expand_dims(np.array(latents[orig_pic]), axis=0)
    latent_code_init = torch.tensor(latent_code).cuda()
    deltas_path = 'invimg/results/weight_deltas/' + orig_pic.split('.')[0] + '.npy'
    deltas = np.load(deltas_path, allow_pickle=True)
    deltas = [torch.from_numpy(w).cuda() if w is not None else None for w in deltas]

    os.makedirs(opts.results, exist_ok=True)


    gan_generator = get_ganmodel(opts)

    # 生成初始图片
    with torch.no_grad():
        inv_img, _ = gan_generator([latent_code_init], input_is_latent=True, randomize_noise=True, weights_deltas=deltas)


    latent = latent_code_init.clone().detach()
    latent.requires_grad = True

    clip_loss = CLIPLoss(opts)
    id_loss = IDLoss(opts)
    optimizer = optim.Adam([latent], lr=opts.alpha)

    pbar = tqdm(range(opts.step))
    for i in pbar:
        t = i / opts.step
        lr = get_lr(t, opts.alpha)
        optimizer.param_groups[0]["lr"] = lr
        img_gen, _ = gan_generator([latent], input_is_latent=True, randomize_noise=True, weights_deltas=deltas)
        c_loss = clip_loss(img_gen, edit_text)
        if opts.id_lambda > 0:
            i_loss = id_loss(img_gen, inv_img)[0]
        else:
            i_loss = 0   # 不需要idloss就不跑模型了，节省时间

        latent_loss = ((latent_code_init - latent) ** 2).sum()

        loss = c_loss + opts.latent_lambda * latent_loss + opts.id_lambda * i_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(
            (
                f"loss: {loss.item():.4f};"
            )
        )
        if opts.save_intermediate_image_every > 0 and i % opts.save_intermediate_image_every == 0:
            with torch.no_grad():
                img_gen, _ = gan_generator([latent], input_is_latent=True, randomize_noise=True)

            torchvision.utils.save_image(img_gen, f"results/{str(i).zfill(5)}.jpg", normalize=True, range=(-1, 1))


        final_result = torch.cat([inv_img, img_gen])
        torchvision.utils.save_image(final_result.detach().cpu(), os.path.join(opts.results, "final_result.jpg"),
                                     normalize=True, scale_each=True, range=(-1, 1))

