1. conda create  -n tmd  python==3.10 
2. conda activate tmd
## module load compiler/gcc/9.1.0 ( if getting : gcc: error: unrecognized command line option ‘-std=c++17’)
3. pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
4. pip install torchmetrics==0.8.2
5. pip install tqdm
6. pip install pytorch-lightning==1.6.3
7. pip install torch-geometric==2.3.0
8. pip install h5py==3.8.0
9. pip install tqdm
10. pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
11. cd mdbenchgnn/models/torchmd-net/
12. pip install --no-deps -e  .
