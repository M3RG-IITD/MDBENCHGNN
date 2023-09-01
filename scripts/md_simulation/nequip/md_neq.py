from nequip.ase import NequIPCalculator
import torch
from ase import units
from ase.md.langevin import Langevin
#from ase.md.nvtberendsen import NVTBerendsen
from ase.io import read, write
from tqdm import tqdm
import argparse

#def main(args): later add argparser
def main(args=None):
    """Run MD simulation with NequIP model
    """
    model_path = args.model_path
    calculator = NequIPCalculator.from_deployed_model(model_path= model_path, device=args.device)
    model_name = "nequip"
    
    for i in tqdm(range(args.init_conf_N)):
        init_conf = read(args.init_conf_path, str(i))
        init_conf.set_calculator(calculator)
        dyn = Langevin(init_conf, args.timestep*units.fs, temperature_K=args.temp, friction=5e-3) 
        def write_frame():
            dyn.atoms.write(args.out_dir + f'md_{model_name}_langevin{i}.xyz', append=True)
        dyn.attach(write_frame, interval=1)
        dyn.run(args.runsteps)
        print(f"MD finished!{i}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MD simulation with NequIP model")
    parser.add_argument("--model_path", type=str, default="example/lips20/nequip/deployed_model.pth", help="Path to the model")
    parser.add_argument("--init_conf_path", type=str, default="example/lips20/data/test/botnet.xyz", help="Path to the initial configuration")
    parser.add_argument("--device", type=str, default="cude", help="Device:["cpu","cuda"]")
    parser.add_argument("--init_conf_N", type=int, default=10, help="No. of initial configuration, [i=0 to N-1] confs read")
    parser.add_argument("--out_dir", type=str, default="out_dir_sl/neqip/lips20/", help="Output path")
    parser.add_argument("--temp", type=float, default=520, help="Temperature in Kelvin")
    parser.add_argument("--timestep", type=float, default=1.0, help="Timestep in fs units")
    parser.add_argument("--runsteps", type=int, default=1000, help="No. of steps in to run")
    
    args = parser.parse_args()
    main(args)
