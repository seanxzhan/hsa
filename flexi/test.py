import os
import argparse
from util import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='flexicubes optimization')
    parser.add_argument('-o', '--out_dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)    
    
    parser.add_argument('-i', '--iter', type=int, default=1000)
    parser.add_argument('-b', '--batch', type=int, default=8)
    parser.add_argument('-r', '--train_res', nargs=2, type=int, default=[2048, 2048])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--voxel_grid_res', type=int, default=64)
    
    parser.add_argument('--sdf_loss', type=bool, default=True)
    parser.add_argument('--develop_reg', type=bool, default=False)
    parser.add_argument('--sdf_regularizer', type=float, default=0.2)
    
    parser.add_argument('-dr', '--display_res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-si', '--save_interval', type=int, default=20)
    FLAGS = parser.parse_args()
    device = 'cuda'
    
    # os.makedirs(FLAGS.out_dir, exist_ok=True)
    glctx = dr.RasterizeGLContext()