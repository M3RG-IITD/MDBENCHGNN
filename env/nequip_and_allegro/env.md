module load compiler/gcc/6.5.0/compilervars

1. conda create -n neq_alg python==3.9
2. conda activate neq_alg
3. conda install  cudatoolkit=11.1  -c conda-forge
4. pip install torch==1.10.1+cu111  -f https://download.pytorch.org/whl/cu111/torch_stable.html
5. pip install wandb
6. pip install --no-deps -r env/nequip_and_allegro/requirements.txt
7. cd mdbenchgnn/models/nequip
8. pip install --no-deps -e . 
9. cd mdbenchgnn/models/allegro
10. pip install --no-deps -e . 
11. pip install python-dateutil



note: it is recommended to use "pip install --no-deps [package_name]" to avoid installation of conflicting dependencies

if getting module not found error please check the version in neq_alg.yml and install it.
example: I got: module not found :torch-runstats (check the version of torch-runstats in neq_alg.yml, it is 0.2.0)
pip install --no-deps torch-runstats==0.2.0


