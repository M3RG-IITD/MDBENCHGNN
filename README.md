# Setup environment
see : [env](env)

# Download example data and model weights(~450 MB)
cd ../
pip install gdown (install gdown to download data from google drive)
gdown --folder --id 1PrrKaMBbjMyt3DrXM94XQ-X48nVrf-kY
ln -s ../example_mdbenchgnn/  example


create couple of softlinks (softlinks are great way to save space and avoid copying data)
cd mdbenchgnn/
ln -s ../example_ example
ln -s  path/to/output_dir output_dir_sl
ln -s  path/to/data_dir data_ls


# Models supported
1. Nequip

2. Allegro

3. Mace

4. Botnet

5. Equiformer

6. TorchMDnet

# Try on example data: [example/lips/data](example/lips/data)
1. to train the models go through: [scripts/train](scripts/train)
2. to run a MD simulation on trained model: [scripts/train](scripts/md_simulation)



# Run on custom datasets:
see : [preprocessing](preprocessing)



# Postprocess

## Acknowledgement ##

Our implementation is based on [PyTorch](https://pytorch.org/), [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html), [nequip](https://github.com/mir-group/nequip/), [allegro](https://github.com/mir-group/allegro), [equiformer](https://github.com/atomicarchitects/equiformer), [mace](https://github.com/utkarshp1161/mace), [TorchMD-NET](https://github.com/torchmd/torchmd-net) and [MDsim](https://github.com/kyonofx/MDsim/)
