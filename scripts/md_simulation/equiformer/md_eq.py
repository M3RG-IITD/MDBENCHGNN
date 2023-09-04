import torch
from ase import units
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.io import read, write

import sys


#sys.path.append('mdbenchgnn/models/equiformer') #Relative to MDBNENCHGNN Repository folder
sys.path.append('./mdbenchgnn/utils/equiformer_ase') #Relative to MDBNENCHGNN Repository folder
sys.path.append('./mdbenchgnn/models/equiformer/') #Relative to MDBNENCHGNN Repository folder

from equiformer_calc import *
import nets
from nets import model_entrypoint
from tqdm import tqdm
import time
import argparse

def main(args):
    start_time=time.time()
    device = args.device
    
    # get from equiformer log
    args = {"input_irreps" : '64x0e' , "radius" : 5.0, "num_basis" : 32, "drop_path" : 0.0 }
    mean = -406737.4375
    std =  5.786451816558838
    model_name = 'graph_attention_transformer_nonlinear_exp_l2_md17'
    
    ''' Network '''
    create_model = model_entrypoint(model_name)

    model = create_model(irreps_in=args["input_irreps"], 
        radius=args["radius"], 
        num_basis=args["num_basis"], 
        task_mean=mean, 
        task_std=std, 
        atomref=None,
        drop_path=args["drop_path"])


    state_dict_path = args.model_path

    state_dict = torch.load(state_dict_path , map_location=args.device)

    model.load_state_dict(state_dict['state_dict'])
    calculator = EquiformerCalculator(model=model, device=args.device, r_max = args["radius"])
    
    model_name = "equiformer"
    #system = "lips"
    
    out_path = "./"
    for i in tqdm(range(10)):
        init_conf = read(args.init_conf_path, str(i))
        init_conf.set_calculator(calculator)
        dyn = Langevin(init_conf, args.timestep*units.fs, temperature_K=args.temp, friction=5e-3)
        def write_frame():
            dyn.atoms.write(args.out_dir + f'md_{model_name}_langevin{i}.xyz', append=True)
        dyn.attach(write_frame, interval=1)
        dyn.run(args.runsteps)
        print(f"MD finished!{i}")
    running_time_seconds = time.time() - start_time
    print("Time taken: "+str(running_time_seconds/60) +" minute ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MD simulation with equiformer model")
    parser.add_argument("--model_path", type=str, default="example/lips20/nequip/deployed_model.pth", help="Path to the model")
    parser.add_argument("--init_conf_path", type=str, default="example/lips20/data/test/botnet.xyz", help="Path to the initial configuration")
    parser.add_argument("--device", type=str, default="cuda", help="Device:['cpu','cuda']")
    parser.add_argument("--init_conf_N", type=int, default=10, help="No. of initial configuration, [i=0 to N-1] confs read")
    parser.add_argument("--out_dir", type=str, default="out_dir_sl/neqip/lips20/", help="Output path")
    parser.add_argument("--temp", type=float, default=520, help="Temperature in Kelvin")
    parser.add_argument("--timestep", type=float, default=1.0, help="Timestep in fs units")
    parser.add_argument("--runsteps", type=int, default=1000, help="No. of steps in to run")
    
    args = parser.parse_args()
    main(args)
