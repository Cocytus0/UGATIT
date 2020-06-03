from UGATIT import UGATIT
import argparse
from utils import *

"""parsing and configuration"""

def parse_args():
    desc = "Tensorflow implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)  
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')  #运行模式，train或test
    parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / U-GAT-IT light version]') #是否使用轻量级模型
    parser.add_argument('--dataset', type=str, default='selfie2anime', help='dataset_name') #使用的数据集名称

    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run') #训练轮数
    parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations') #迭代次数
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')  #batch大小
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')  #图片打印频率
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq') #checkpoint保存频率
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_epoch', type=int, default=50, help='decay epoch')  #可能是指的学习率衰减相关参数

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')  #学习率
    parser.add_argument('--GP_ld', type=int, default=10, help='The gradient penalty lambda') #梯度惩罚， 是WGAN-GP相关
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight about GAN') #gan权重参数
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight about Cycle') #cycle权重参数
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight about Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight about CAM')  #CAM注意力机制权重参数
    parser.add_argument('--gan_type', type=str, default='lsgan', help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge]') #gan的类型

    parser.add_argument('--smoothing', type=str2bool, default=True, help='AdaLIN smoothing effect') #是否使用AdaLIN平滑（猜测）

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')   #每层的通道数
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')  #残差块数量
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer') #discriminator的层数
    parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')  #不清楚
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm') #谱归一化，效果和GP还有WC类似，皆为满足1-Lipschitz
    
    parser.add_argument('--img_size', type=int, default=256, help='The size of image') #图片的大小
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel') #图片的通道数
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not') #是否进行图像增强（具体不知）

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')   #保存checkpoint的文件夹
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images') #保存生成图像的文件夹
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')  #保存训练时的sample的文件夹

    return check_args(parser.parse_args())

"""checking arguments"""  //检查参数是否设置非法
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""   #主函数
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = UGATIT(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

if __name__ == '__main__':
    main()
