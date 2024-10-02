conda create --name kd python=3.8 -y
conda activate kd

conda install -c "nvidia/label/cuda-11.3.1" cuda
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install tqdm
pip install transformers
pip install scipy
pip install pandas
pip install torchsummary
pip install nvitop

pip install accelerate

pip install simple-gpu-scheduler
pip install matplotlib
pip install scikit-learn

conda install jupyterlab -y
