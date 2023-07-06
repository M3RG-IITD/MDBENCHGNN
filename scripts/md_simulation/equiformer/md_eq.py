from mdbenchgnn.utils.equiformer_ase.equiformer_calc import EquiformerCalculator
import torch
from ase import units
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.io import read, write

import sys
sys.path.append('mdbenchgnn/models/equiformer')
import nets
from nets import model_entrypoint
from tqdm import tqdm

def main():
    device = "cuda"
    
    # get from equiformer log
    args = {"input_irreps" : '64x0e' , "radius" : 5.0, "num_basis" : 32, "drop_path" : 0.0 }
    mean = -357.6152648925781
    std = 0.5507857203483582
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


    state_dict_path = "example/lips/equiformer/best_val_epochs@74_e@0.0833_f@0.0510.pth.tar"

    state_dict = torch.load(state_dict_path , map_location="cuda")

    model.load_state_dict(state_dict['state_dict'])
    calculator = EquiformerCalculator(model=model, device='cuda', r_max = args["radius"])
    
    model_name = "equiformer"
    system = "lips"
    
    out_path = "./"
    for i in tqdm(range(10)):
        init_conf = read('example/lips/data/test/botnet.xyz', str(i))
        init_conf.set_calculator(calculator)
        dyn = Langevin(init_conf, 2*units.fs, temperature_K=520, friction=5e-3)
        def write_frame():
            dyn.atoms.write(out_path + f'md_{system}_{model_name}_langevin{i}.xyz', append=True)
        dyn.attach(write_frame, interval=1)
        dyn.run(1000)
        print(f"MD finished!{i}")

if __name__ == "__main__":
    main()
