### 运行hyperstyle的命令行

python scripts/inference.py --exp_dir=results --checkpoint_path=pretrained_models/hyperstyle_ffhq.pt --data_path=input_imgs/ --test_batch_size=4 --test_workers=4 --n_iters_per_batch=5 --load_w_encoder --w_encoder_checkpoint_path pretrained_models/faces_w_encoder.pt --save_weight_deltas



### FFHQ数据集

[NVlabs/ffhq-dataset: Flickr-Faces-HQ Dataset (FFHQ) (github.com)](https://github.com/NVlabs/ffhq-dataset)



### pycharm运行时的问题

pycharm 不能运行Ninjia命令行可以 [Ninja is required to load C++ extensions in Pycharm-爱代码爱编程 (icode.best)](https://icode.best/i/69588545838995)

hyperstyle子文件库不能import， 添加interpret path[pycharm 添加python path_王浩的专栏-程序员信息网_pycharm 设置pythonpath - 程序员信息网 (i4k.xyz)](https://www.i4k.xyz/article/wh357589873/53204024)

解决can't find model的问题：添加_init_.py文件，utils换个名字，使用绝对Import

[How can I fix "No module named 'models'" error? (More details in post) : learnpython (reddit.com)](https://www.reddit.com/r/learnpython/comments/r53vnf/how_can_i_fix_no_module_named_models_error_more/)

[Python 模块 | 菜鸟教程 (runoob.com)](https://www.runoob.com/python/python-modules.html)

### argparser

[argparse模块用法实例详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/56922793)

### 一些tensor操作

#### .eval()

如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。

训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有BN层和Dropout所带来的的性质。

在做one classification的时候，训练集和测试集的样本分布是不一样的，尤其需要注意这一点。

#### clone和detach

[(4条消息) 【Pytorch】对比clone、detach以及copy_等张量复制操作_guofei_fly的博客-CSDN博客_pytorch 复制张量](https://blog.csdn.net/guofei_fly/article/details/104486708)



### 图片读入后转化为归一化tensor

[(4条消息) Pytorch：torchvision.transforms_宁静致远*的博客-CSDN博客](https://blog.csdn.net/weixin_40522801/article/details/106037353)

[数据预处理中的归一化与反归一化_TracelessLe的专栏-程序员资料_归一化和反归一化公式 - 程序员资料 (4k8k.xyz)](http://www.4k8k.xyz/article/TracelessLe/116021329)

### 五官的实例分割

使用[zllrunning/face-parsing.PyTorch: Using modified BiSeNet for face parsing in PyTorch (github.com)](https://github.com/zllrunning/face-parsing.PyTorch)完成实例分割。

原方法同时分割许多类，划分非常精细，如左眼右眼，上唇下唇。我们写一块小代码，获取用户指定的部分，如[左眼、右眼、鼻子]，将每个指定部分变成白色，其他全部变成黑色。

![image-20220502140221231](https://cdn.jsdelivr.net/gh/wangyuchi369/picbed/img/202205021404629.png)

### 添加Mask

利用上面实体分割添加指定部分的mask。

计算mask二范数时直接将图像差点乘mask tensor即可，因为都是0和1，mask区域的面积可以直接用一范数

![image-20220502204043760](https://cdn.jsdelivr.net/gh/wangyuchi369/picbed/img/202205022040397.png)


### Jupyter 无法处理命令行


A more elegant solution would be:

```args, unknown = parser.parse_known_args()```

instead of

```args = parser.parse_args()```

[python - How to fix ipykernel_launcher.py: error: unrecognized arguments in jupyter? - Stack Overflow](https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter)