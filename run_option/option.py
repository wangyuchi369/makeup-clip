import argparse
from invimg.configs.paths_config import model_paths
class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='控制参数')

    def get_args(self):
        """
        初始化参数
        Returns: parser
        """
        self.parser.add_argument("--gan_model", type=str,
                                 default="optimclip/pretrained_models/stylegan2-ffhq-config-f.pt",
                                 help="预训练的stylegan模型")
        self.parser.add_argument("--size", type=int, default=1024, help="图片分辨率")
        self.parser.add_argument("--alpha", type=float, default=0.1, help='初始学习率')
        self.parser.add_argument("--step", type=int, default=30, help="迭代次数")
        self.parser.add_argument("--latent_lambda", type=float, default=0.008,
                                 help="latent-code损失的系数")
        self.parser.add_argument("--img_lambda", type=float, default=0, help="图片损失的系数")
        self.parser.add_argument("--id_lambda", type=float, default=0.001, help="面部损失的系数")
        self.parser.add_argument("--results", type=str, default='result/opt/', help="结果放置的文件夹")
        self.parser.add_argument('--id_model', default='optimclip/pretrained_models/model_ir_se50.pth', type=str,
                                 help="图像识别网络")
        self.parser.add_argument("--save_intermediate_image_every", type=int, default=20,
                                 help="每隔一定步数保存结果")
        # self.parser.add_argument("--bbox", type=list, default=[413,537,254,749],
        #                          help="部位位置（上下左右的顺序）")
        self.parser.add_argument('--exp_dir', type=str, default='result/inv',
                                 help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_path', default='invimg/pretrained_models/hyperstyle_ffhq.pt', type=str,
                                 help='Path to HyperStyle model checkpoint')
        self.parser.add_argument('--data_path', type=str, default='input_img/',
                                 help='Path to directory of images to evaluate')
        self.parser.add_argument('--resize_outputs', action='store_true',
                                 help='Whether to resize outputs to 256x256 or keep at original output resolution')
        self.parser.add_argument('--test_batch_size', default=2, type=int,
                                 help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')
        self.parser.add_argument('--n_images', type=int, default=None,
                                 help='Number of images to output. If None, invert on all data')
        self.parser.add_argument('--save_weight_deltas', action='store_true', default=True,
                                 help='Whether to save the weight deltas of each image. Note: file weighs about 200MB.')

        # arguments for iterative inference
        self.parser.add_argument('--n_iters_per_batch', default=5, type=int,
                                 help='Number of forward passes per batch during training.')

        # arguments for loading pre-trained encoder
        self.parser.add_argument('--load_w_encoder', action='store_true', default=True,
                                 help='Whether to load the w e4e encoder.')
        self.parser.add_argument('--w_encoder_checkpoint_path', default='invimg/pretrained_models/faces_w_encoder.pt',
                                 type=str,
                                 help='Path to pre-trained W-encoder.')
        self.parser.add_argument('--w_encoder_type', default='WEncoder',
                                 help='Encoder type for the encoder used to get the initial inversion')

        # arguments for editing scripts
        self.parser.add_argument('--edit_directions', default='age,smile,pose',
                                 help='which edit directions top perform')
        self.parser.add_argument('--factor_range', type=int, default=5, help='max range for interfacegan edits.')

        # arguments for domain adaptation
        self.parser.add_argument('--restyle_checkpoint_path', default=model_paths["restyle_e4e_ffhq"], type=str,
                                 help='ReStyle e4e checkpoint path used for domain adaptation')
        self.parser.add_argument('--restyle_n_iterations', default=2, type=int,
                                 help='Number of forward passes per batch for ReStyle-e4e inference.')
        self.parser.add_argument('--finetuned_generator_checkpoint_path', type=str,
                                 default=model_paths["stylegan_pixar"],
                                 help='Path to fine-tuned generator checkpoint used for domain adaptation.')
        return self.parser.parse_known_args()[0]