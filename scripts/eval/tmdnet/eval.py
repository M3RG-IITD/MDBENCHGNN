from torchmdnet.models.model import load_model
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import argparse

import torch
import ase.data
import ase.io

"""
Example command:

python "scripts/eval/tmdnet/eval.py" \
--model "/home/civil/btech/ce1180169/Output_mdbgnn/tmdnet/md17/aspirin/epoch=2739-val_loss=0.1136-test_loss=0.1902.ckpt" \
--device 'cuda' \
--data_path "/home/civil/btech/ce1180169/MDBENCHGNN/example_mdbenchgnn/md17/aspirin/test/botnet.xyz"

"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to model checkpoint", required=True)
    parser.add_argument("--device",help="select device",type=str,choices=['cpu', 'cuda'], default="cuda")
    parser.add_argument("--data_path", help="Path to configurations data xyz format", type=str, required=True)
    
    
    return parser.parse_args()


def main():
    
    args = parse_args()
    model = load_model(args.model, derivative = True)
    atoms_list = ase.io.read(args.data_path, index=":")
    
    
    y_true_e = []
    y_pred_e = []
    y_true_f=[]
    y_pred_f=[]
    for i in tqdm(range(len(atoms_list))):# 10% sample
        sample_an = torch.tensor(atoms_list[i].get_atomic_numbers(), dtype=torch.int64)
        sample_pos = torch.tensor(atoms_list[i].get_positions(), dtype = torch.float)
        pred_e , pred_f = model(sample_an, sample_pos)
        y_pred_e.append(pred_e.detach().numpy()[0][0])
        y_true_e.append(atoms_list[i].get_total_energy())
        
        y_pred_f.append(pred_f.detach().numpy().reshape(-1).tolist())
        y_true_f.append(atoms_list[i].get_forces().reshape(-1).tolist())

    
    E_mae=mean_absolute_error(y_true_e, y_pred_e)
    F_mae=mean_absolute_error(y_true_f, y_pred_f)
    print("||Final Results:||")
    print("E_MAE: ", round(E_mae,3) ,"\t F_MAE: ",round(F_mae.item(),3) )
    
if __name__ == "__main__":
    main()
