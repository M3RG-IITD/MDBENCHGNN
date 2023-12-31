Code to paper: [EGRAFFBENCH: EVALUATION OF EQUIVARIANT GRAPH
NEURAL NETWORK FORCE FIELDS FOR ATOMISTIC
SIMULATIONS](https://arxiv.org/pdf/2310.02428.pdf)


<div style="text-align:center">
  <img src="assets/mdbenchgnn_logo.jpeg" alt="Image" width="350" height="350">
</div>




# Setup environment
see : [env](env)

# Download example data and model weights(~450 MB): folder example_mdbenchgnn 
1. cd ../
2. pip install gdown (install gdown to download data from google drive)
3. gdown --folder --id 1PrrKaMBbjMyt3DrXM94XQ-X48nVrf-kY




# create few symbolic-links (symbolic-links are great way to save space and avoid copying data)
1. cd mdbenchgnn/
2. ln -s ../example_mdbenchgnn example
3. ln -s  path/to/output_dir output_dir_sl
4. ln -s  path/to/data_dir data_ls


# Models supported
1. Nequip

2. Allegro

3. Mace

4. Botnet

5. Equiformer

6. TorchMDnet

# Try on example data: [example/lips/data](example/lips/data)
1. to train the models go through: [scripts/train](scripts/train)
2. to run a MD simulation on trained model: [scripts/md_simulation](scripts/md_simulation)



# Run on custom datasets:
see : [preprocessing](preprocessing)



# Postprocess
1. Radial Distribution function [scripts/struct_props](scripts/struct_props)

## Acknowledgement ##

Our implementation is based on [PyTorch](https://pytorch.org/), [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html), [nequip](https://github.com/mir-group/nequip/), [allegro](https://github.com/mir-group/allegro), [equiformer](https://github.com/atomicarchitects/equiformer), [mace](https://github.com/utkarshp1161/mace), [TorchMD-NET](https://github.com/torchmd/torchmd-net) and [MDsim](https://github.com/kyonofx/MDsim/)
