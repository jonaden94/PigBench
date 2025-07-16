# when running this setup CUDA 11.8 and GCC 11.4.0 must be available
ENV_NAME='pigbench'
conda env remove -n $ENV_NAME -y
conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

############# torch
# module load cuda/11.8 # Lmod command to load cuda version compatible with the installed pytorch version
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y # fixes some cudnn loading error
conda install mkl=2024.0.0 -y # fixes some pytorch bug

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

############################################ COMMENT OUT EVERYTHING BELOW IF YOU DO NOT WANT TO USE MOTRv2 AND MOTIP.
############# motr deformable attention
# !! NEED TO MAKE SURE CUDA IS AVAILABLE AND VISIBLE TO THE COMPILER THROUGH CUDA_VISIBLE_DEVICES
# !! NEED TO MAKE SURE GCC VERSION IS COMPATIBLE WITH CUDA VERSION. IN COMPUTE CLUSTER YOU CAN USUALLY MANUALLY LOAD CUDA/GCC MODULES
# module load gcc/11.4.0-cuda # Lmod command to load gcc version that is compatible with cuda version
cd tracking/motrv2/models/ops
sh make.sh

############# motip deformable attention 
# !! NEED TO MAKE SURE CUDA IS AVAILABLE AND VISIBLE TO THE COMPILER THROUGH CUDA_VISIBLE_DEVICES
# !! NEED TO MAKE SURE GCC VERSION IS COMPATIBLE WITH CUDA VERSION. IN COMPUTE CLUSTER YOU CAN USUALLY MANUALLY LOAD CUDA/GCC MODULES
# module load gcc/11.4.0-cuda # Lmod command to load gcc version that is compatible with cuda version
cd ../../../motip/models/ops
sh make.sh
cd ../../..

conda deactivate