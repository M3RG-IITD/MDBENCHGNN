# Create a virtual environment and activate it
conda create --name ma_bo
conda activate ma_bo

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# (optional) Install MACE's dependencies from Conda as well
conda install numpy scipy matplotlib ase opt_einsum prettytable pandas e3nn

cd mdbenchgnn/models/mace
pip install  -e . 

if getting module not found error please check the version in ma_bo.yml and install it.

