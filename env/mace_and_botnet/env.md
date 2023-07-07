# Create a virtual environment and activate it
1. conda create --name ma_bo
2. conda activate ma_bo
3. conda install python==3.9.12
4. pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
5. cd mdbenchgnn/models/mace
6. pip install  -e . 

## if getting module not found error please check the version in ma_bo.yml and install it.

