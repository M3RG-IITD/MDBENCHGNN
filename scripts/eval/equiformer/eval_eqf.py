import torch
from ase import units
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.io import read, write
import ase.data
import ase.io

import sys


#sys.path.append('mdbenchgnn/models/equiformer') #Relative to MDBNENCHGNN Repository folder
sys.path.append('./mdbenchgnn/utils/equiformer_ase') #Relative to MDBNENCHGNN Repository folder
sys.path.append('./mdbenchgnn/models/equiformer/') #Relative to MDBNENCHGNN Repository folder
sys.path.append("mdbenchgnn/models/ocp/")

import nets
from nets import model_entrypoint
from tqdm import tqdm
import time
import argparse


"""
1)Use eqf environent
2){"input_irreps" : '64x0e' , "radius" : 5.0, "num_basis" : 32, "drop_path" : 0.0}
3}Provide mean and std from the log file

Example command:


python "scripts/eval/equiformer/eval_eqf.py" \
--model "/home/civil/btech/ce1180169/Output_mdbgnn/equiformer/md17/aspirin/epochs@4899_e@0.1360_f@0.3024.pth.tar" \
--device 'cpu' \
--mean -406737.4375 \
--std 5.786451816558838 \
--data_path "/home/civil/btech/ce1180169/MDBENCHGNN/example_mdbenchgnn/md17/aspirin/test/botnet.xyz"


"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--mean", type=float,help="mean", required=True)
    parser.add_argument("--std", type=float,help="std", required=True)
    
    #parser.add_argument("--args_log",default= {"input_irreps" : '64x0e' , "radius" : 5.0, "num_basis" : 32, "drop_path" : 0.0,"mean" : -406737.4375,"std" : 5.786451816558838},required=True)
    parser.add_argument("--device",help="select device",type=str,choices=['cpu', 'cuda'], default="cuda")
    parser.add_argument("--data_path", help="Path to configurations data xyz format", type=str, required=True)
    
    
    return parser.parse_args()


def main():
    
    args = parse_args()
    
    # get from equiformer log
    args_log = {"input_irreps" : '64x0e' , "radius" : 5.0, "num_basis" : 32, "drop_path" : 0.0,"mean":args.mean,"std" :  args.std}
    model_name = 'graph_attention_transformer_nonlinear_exp_l2_md17'
    
    print("Loading model....")
    ''' Network '''
    create_model = model_entrypoint(model_name)

    model = create_model(irreps_in=args_log["input_irreps"], 
        radius=args_log["radius"], 
        num_basis=args_log["num_basis"], 
        task_mean=args_log["mean"], 
        task_std=args_log["std"], 
        atomref=None,
        drop_path=args_log["drop_path"])


    state_dict_path = args.model

    state_dict = torch.load(state_dict_path , map_location=args.device)

    model.load_state_dict(state_dict['state_dict'])
    #model.to(args.device)
    print("Model Loaded!")
    
    
    atoms_list = ase.io.read(args.data_path, index=":")
    
    y_true_e = []
    y_pred_e = []
    y_true_f=[]
    y_pred_f=[]
    counter=0
    e_mae=0
    f_mae=0
    for i in range(len(atoms_list)):# 10% sample
        counter+=1
        data=atoms_list[i]
        
        sample_atm_nos=torch.Tensor(data.get_atomic_numbers()).long().reshape(-1).to(args.device)
        sample_pos=torch.Tensor(data.get_positions()).to(args.device)
        with torch.no_grad():
            pred_e, pred_f = model(node_atom=sample_atm_nos, pos=sample_pos, batch=torch.zeros_like(sample_atm_nos,dtype=torch.int64).to(args.device))
        pred_e=pred_e*args.std+args.mean
        pred_f=pred_f * args.std
        #y_pred_e.append(pred_e.detach().numpy()[0][0])
        #y_true_e.append(atoms_list[i].get_total_energy())
        
        #y_pred_f.append(pred_f.detach().numpy().reshape(-1).tolist())
        #y_true_f.append(atoms_list[i].get_forces().reshape(-1).tolist())
        temp_e=(abs(pred_e.detach().numpy()-atoms_list[i].get_total_energy())).mean()
        temp_f=(abs(pred_f.detach().numpy()-atoms_list[i].get_forces())).mean()
        print("Batch: ",counter,"\te_mae: ",round(temp_e.item(),3),"\tf_mae: ",round(temp_f.item(),3))
        e_mae+=temp_e
        f_mae+=temp_f
    print("||Final Results:||")
    print("E_MAE: ", round((e_mae/counter).item(),3) ,"\t F_MAE: ",round((f_mae/(counter)).item(),3) )

    
    #E_mae=mean_absolute_error(y_true_e, y_pred_e)
    #F_mae=mean_absolute_error(np.array(y_true_f), np.array(y_pred_f))
    #print("||Final Results:||")
    #print("E_MAE: ", round(E_mae,3) ,"\t F_MAE: ",round(F_mae.item(),3) )
    
if __name__ == "__main__":
    main()
