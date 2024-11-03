# Set up python
conda create -n hydragen python=3.11.5
conda activate hydragen
# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia # TODO: Consider upgrading the Python version
conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y nvidia/label/cuda-12.1.0::cuda

pip install packaging==24.0 ninja==1.11.1.1 && pip install -e .

pip install -r requirements-dev.txt
