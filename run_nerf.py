import torch
import numpy as np
import os, sys

import configargparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='',
                        help='input data directory')
    parser.add_argument('--videodir', type=str, default='./data/ingp/ukulele/ukulele.mp4',
                        help='input video directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=1,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--i_embed_views", type=int, default=2,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')
    # logging/saving options
    parser.add_argument("--n_iters", type=int, default=3000,
                        help='number of iteration')
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=3000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=3000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=30000000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=3000000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--i_mesh", type=int, default=3000,
                        help='frequency of mesh saving')

    parser.add_argument("--finest_res", type=int, default=512,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--log2_hashmap_size", type=int, default=19,
                        help='log2 of hashmap size')
    parser.add_argument("--sparse-loss-weight", type=float, default=1e-10,
                        help='learning rate')
    parser.add_argument("--tv-loss-weight", type=float, default=1e-6,
                        help='learning rate')

    parser.add_argument("--video_in", type=str, default="",
                        help='./video/path/video.mp4')

    # mesh options
    parser.add_argument("--mesh_only", action='store_true',
                        help='do not optimize, reload weights and generate mesh')
    parser.add_argument("--mesh_res", type=int, default=256,
                        help='resolution of grid for marching cubes')

    # save loss psnr graph
    parser.add_argument("--train_graph", action='store_true',
                        help='Check loss and psnr during training.')

    return parser

def train():
    # set a configuration parser
    parser = config_parser()
    args = parser.parse_args()

    # Load data


    return

if __name__=='__main__':
    # Set the default tensor type to CUDA (GPU) FloatTensor
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    train()

