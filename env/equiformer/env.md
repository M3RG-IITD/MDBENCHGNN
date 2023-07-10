conda env create -f eqf_env.yml
conda activate eqf

pip install e3nn==0.4.4
pip install git+https://github.com/Open-Catalyst-Project/ocp@b5a197f
pip  install matplotlib==3.3.4
pip install lmdb==1.1.1