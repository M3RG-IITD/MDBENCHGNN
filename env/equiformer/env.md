## module load compiler/gcc/6.5.0/compilervars ( if getting : gcc: error: unrecognized command line option ‘-std=c++17’)
conda env create -f eqf_env.yml
conda activate eqf
cd mdbenchgnn/models/ocp
git checkout b5a197f
pip install -e .

