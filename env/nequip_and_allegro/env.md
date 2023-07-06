conda create -n neq_alg python==3.9
conda activate neq_alg
conda install  cudatoolkit=11.1  -c conda-forge
pip install torch==1.10.1+cu111  -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install wandb

note: it is recommended to use "pip install --no-deps [package_name]" to avoid installation of conflicting dependencies

conda activate neq_alg
cd mdbenchgnn/models/nequip
pip install --no-deps -e . 
cd mdbenchgnn/models/allegro
pip install --no-deps -e . 

if getting module not found error please check the version in neq_alg.yml and install it.
example: I got: module not found :torch-runstats (check the version of torch-runstats in neq_alg.yml, it is 0.2.0)
pip install --no-deps torch-runstats==0.2.0