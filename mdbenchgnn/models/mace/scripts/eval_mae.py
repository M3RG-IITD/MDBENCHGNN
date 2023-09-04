###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model for 
# energy and force MAE
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse

import ase.data
import ase.io
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, utils, torch_tools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--output",help="output path",required=True)
    parser.add_argument("--device",help="select device",type=str,choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--default_dtype",help="set default dtype",type=str,choices=["float32", "float64"],default="float32")
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("--info_prefix",help="prefix for energy, forces and stress keys",type=str,default="MACE_")
    
    
    return parser.parse_args()


def main():
    args = parse_args()
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=args.device)

    # Load data and prepare input
    atoms_list = ase.io.read(args.configs, index=":")
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max))
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    
    #Create counter variables
    counter=0
    e_mae=0
    f_mae=0
    
    for batch in data_loader:
        counter+=args.batch_size
        batch = batch.to(device)
        output = model(batch.to_dict())
        temp_e=(abs(batch['energy']-output['energy'])).sum()
        temp_f=(abs(batch['forces']-output['forces'])).sum()
        print("Batch: ",counter,"\te_mae: ",round(temp_e.item()/args.batch_size,3),"\tf_mae: ",round(temp_f.item()/args.batch_size,3))
        e_mae+=temp_e
        f_mae+=temp_f
    print("||Final Results:||")
    print("E_MAE: ", round((e_mae/counter).item(),3) ,"\t F_MAE: ",round((f_mae/counter).item(),3) )
    

if __name__ == "__main__":
    main()
