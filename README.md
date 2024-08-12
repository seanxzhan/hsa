```
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.1%2Bcu118.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1%2Bcu118.html
```

flexicubes doesn't work with pyrender, make sure the file where you run flexicubes doesn't import other files containing pyrener

preprocess_3 to get entire graph
write further merge info json
preprocess_17 save_data=False to get further merged graph
peek to get parts of interest
preprocess_17 to get actual data

```
conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html
```

```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install imageio trimesh tqdm matplotlib torch_scatter ninja
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.0_cu113.html
```

OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root
conda install -c conda-forge cudatoolkit-dev

to get nvcc:
```
$ conda install -c nvidia cuda

$ which nvcc
/home/stevekm/miniconda3/envs/nerfstudio/bin/nvcc

$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

<!-- export PATH=~/.conda/envs/flexicubes/include/cuda:$PATH
export LD_LIBRARY_PATH=~/.conda/envs/flexicubes/nvvm/lib64:$LD_LIBRARY_PATH -->

/usr/bin/ld: cannot find -lcudart: No such file or directory
collect2: error: ld returned 1 exit status
ninja: build stopped: subcommand failed.

after `cd .conda/envs/flexicubes`, create `lib64` and copy `cp ./lib/libcudart* ./lib64`