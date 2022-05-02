#!/usr/bin/python
# -*- encoding: utf-8 -*-


from faceparsing.model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

# def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
#     # Colors for all 20 parts
#     part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
#                    [255, 0, 85], [255, 0, 170],
#                    [0, 255, 0], [85, 255, 0], [170, 255, 0],
#                    [0, 255, 85], [0, 255, 170],
#                    [0, 0, 255], [85, 0, 255], [170, 0, 255],
#                    [0, 85, 255], [0, 170, 255],
#                    [255, 255, 0], [255, 255, 85], [255, 255, 170],
#                    [255, 0, 255], [255, 85, 255], [255, 170, 255],
#                    [0, 255, 255], [85, 255, 255], [170, 255, 255]]
#
#     im = np.array(im)
#     vis_im = im.copy().astype(np.uint8)
#     vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
#     vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
#     vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
#
#     num_of_class = np.max(vis_parsing_anno)
#
#     for pi in range(1, num_of_class + 1):
#         index = np.where(vis_parsing_anno == pi)
#         vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
#
#     vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
#     # print(vis_parsing_anno_color.shape, vis_im.shape)
#     vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
#
#     # Save result or not
#     if save_im:
#         cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
#         cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#
#     # return vis_im

atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']


def evaluate(need, result_dir, dspth='./data', cp='model_final_diss.pth'):
    os.makedirs(result_dir, exist_ok=True)
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = cp
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing.shape)
            # print(np.unique(parsing))
            # print(np.array(atts)[np.unique(parsing)])
            # print(image_path)
            create_mask(parsing, need, os.path.join(result_dir, image_path))


            # vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))

def create_mask(parsing, need, result_dir):
    new_parsing = np.ones_like(parsing) * 255
    for each_att in need:
        index = np.where(parsing == atts.index(each_att) + 1)
        # print(index)
        new_parsing[index] = 0
    # print(new_parsing)
    new_parsing = cv2.resize(new_parsing.astype('float32'), (1024, 1024), interpolation=cv2.INTER_LINEAR)
    # print(new_parsing.shape)
    cv2.imwrite(result_dir, new_parsing)







if __name__ == "__main__":
    evaluate(need=['skin'], result_dir='./result.png', dspth='./input_img/', cp='79999_iter.pth')


