export CONDA_ALWAYS_YES="true"

# Set up python
conda create -n hydragen python=3.11.5
conda activate hydragen
# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia # NOTE: Old versions
conda install pytorch==2.4.0 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
# conda install nvidia/label/cuda-12.1.0::cuda # NOTE: asdf
conda install nvidia/label/cuda-12.4.0::cuda

pip install packaging==24.0 ninja==1.11.1.1 --root-user-action ignore
MAX_JOBS=4 pip install -e . --root-user-action ignore --no-build-isolation

pip install -r requirements-dev.txt --root-user-action ignore

pip install -U "huggingface_hub[cli]"

unset CONDA_ALWAYS_YES
