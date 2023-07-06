conda env create -f tmd_env.yml
conda activate tmd_env.yml
cd mdbenchgnn/models/torchmd-net
pip install --no-deps -e .