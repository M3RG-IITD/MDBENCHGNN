# Create a virtual environment and activate it
conda create --name ma_bo
conda activate ma_bo
conda install python==3.9.12

# Install PyTorch
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

cd mdbenchgnn/models/mace
pip install  -e . 

if getting module not found error please check the version in ma_bo.yml and install it.

