from invimg.scripts.inference import invert
import math
import os

import torch
import torchvision
from tqdm import tqdm
import numpy as np
from optimclip.criteria.clip_loss import CLIPLoss
from optimclip.criteria.id_loss import IDLoss
from optimclip.models.stylegan2.model import Generator
import clip
from faceparsing.test import evaluate

from PIL import Image
from torchvision import transforms
from run_option.option import Options


# invert()

def get_ganmodel(opts):
    generator = Generator(opts.size, 512, 8, channel_multiplier=2)
    # TODO 看看generator
    model = torch.load(opts.gan_model)['g_ema']
    generator.load_state_dict(model, strict=True)
    generator = generator.eval().cuda()
    return generator


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def get_init_latent(orig_pic):
    latent_path = 'invimg/results/latents.npy'
    try:
        latents = np.load(latent_path, allow_pickle=True).item()
        latent_code = np.expand_dims(np.array(latents[orig_pic]), axis=0)
    except Exception:
        invert()  # 没有当前图片的latent code，再invert一遍
        latents = np.load(latent_path, allow_pickle=True).item()
        latent_code = np.expand_dims(np.array(latents[orig_pic]), axis=0)
    latent_code_init = torch.tensor(latent_code).cuda()
    deltas_path = 'invimg/results/weight_deltas/' + orig_pic.split('.')[0] + '.npy'
    deltas = np.load(deltas_path, allow_pickle=True)
    deltas = [torch.from_numpy(w).cuda() if w is not None else None for w in deltas]
    return latent_code_init, deltas


def get_imgloss(region, orig_img, img_gen, mask):
    img_loss_sum = torch.sum(torch.square(orig_img - img_gen))
    img_loss = 0
    if region:
        if 'bbox' in region:
            bbox = region['bbox']
            crop_area = (orig_img - img_gen)[:][:][bbox[0]:bbox[1]][bbox[2]:bbox[3]]
            img_loss = img_loss_sum - torch.sum(torch.square(crop_area))
            area = opts.size ** 2 - abs(bbox[0] - bbox[1]) * abs(bbox[2] - bbox[3])  # 剩余的面积
            img_loss /= area
        elif 'organ' in region:
            # print(mask.shape)
            img_loss = torch.sum(torch.square(orig_img * mask - img_gen * mask))
            area = mask.norm(1)  # 1的个数即为他的一范数
            img_loss /= area
        else:
            print('region输入错误')
    else:
        img_loss = img_loss_sum / (opts.size ** 2)
    return img_loss


def optim(text, input_img, opts, region):
    # 分词并拼接
    edit_text = torch.cat([clip.tokenize(text)]).cuda()

    orig_img = Image.open(input_img)
    convert = transforms.ToTensor()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    orig_img = normalize(convert(orig_img))
    orig_img = orig_img.unsqueeze(0).cuda()

    orig_pic = str(input_img).split('/')[-1]
    latent_code_init, deltas = get_init_latent(orig_pic)

    os.makedirs(opts.results, exist_ok=True)
    gan_generator = get_ganmodel(opts)

    # 生成初始图片
    with torch.no_grad():
        inv_img, _ = gan_generator([latent_code_init], input_is_latent=True, randomize_noise=True,
                                   weights_deltas=deltas)

    latent = latent_code_init.clone().detach()
    latent.requires_grad = True

    clip_loss = CLIPLoss(opts)
    id_loss = IDLoss(opts)
    optimizer = torch.optim.Adam([latent], lr=opts.alpha)

    # 得到感兴趣的区域的mask
    mask = None
    if region and 'organ' in region:
        evaluate(region['organ'], 'result/faceparsing/', dspth='input_img/', cp='./faceparsing/res/cp/79999_iter.pth')
        mask = Image.open('result/faceparsing/' + orig_pic)
        mask = convert(mask).cuda()
        mask = mask.repeat(3, 1, 1)
        mask = mask.unsqueeze(0)

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
            i_loss = 0  # 不需要idloss就不跑模型了，节省时间
        latent_loss = ((latent_code_init - latent) ** 2).sum()
        img_loss = get_imgloss(region, orig_img, img_gen, mask)
        # print('latent_loss', latent_loss)
        # print('img_loss', img_loss)
        loss = c_loss + opts.latent_lambda * latent_loss + opts.id_lambda * i_loss + opts.img_lambda * img_loss

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

        final_result = torch.cat([orig_img, inv_img, img_gen, mask])
        torchvision.utils.save_image(final_result.detach().cpu(), os.path.join(opts.results, "final_result.jpg"),
                                     normalize=True, scale_each=True, range=(-1, 1))


if __name__ == '__main__':
    opts = Options().get_args()
    optim(text='a person with purple hair', input_img='input_img/img1.png', opts=opts, region={'organ': ['hair']})
