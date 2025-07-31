# when running this setup CUDA 11.8 and GCC 11.4.0 must be available
ENV_NAME='pigbench'
conda env remove -n $ENV_NAME -y
conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

############# torch
# module load cuda/11.8 # Lmod command to load cuda version compatible with the installed pytorch version
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 mkl=2024.0.0 -c pytorch -c nvidia -y
conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y # fixes some cudnn loading error

############# pip packages
pip install -r _setup/requirements.txt

############# mmdetection/mmyolo
conda install anaconda::lit -y
pip install cmake
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.0"
mim install mmdet==3.3.0
mim install mmyolo==0.6.0

conda deactivate
