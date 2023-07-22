## module load compiler/gcc/6.5.0/compilervars ( if getting : gcc: error: unrecognized command line option ‘-std=c++17’)
1. conda env create -f eqf_env.yml
2. conda activate eqf
3. pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.2+${CUDA}.html
4. cd mdbenchgnn/models/ocp
5. pip install -e .

