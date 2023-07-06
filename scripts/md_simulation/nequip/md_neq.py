from nequip.ase import NequIPCalculator
import torch
from ase import units
from ase.md.langevin import Langevin
#from ase.md.nvtberendsen import NVTBerendsen
from ase.io import read, write
from tqdm import tqdm
import argparse

#def main(args): later add argparser
def main():
    """Run MD simulation with NequIP model
    """
    # model_path = args.model_path
    # temp = args.temp
    # system = args.system
    model_path = "example/lips/nequip/deployed.pth"
    calculator = NequIPCalculator.from_deployed_model(model_path= model_path, device='cuda')
    device = "cuda"; model_name = "nequip"

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
    # parser = argparse.ArgumentParser(description="Run MD simulation with NequIP model")
    # parser.add_argument("--model_path", type=str, default="example/lips/nequip/deployed.pth", help="Path to the model")
    # parser.add_argument("--temp", type=float, default=520, help="Temperature in Kelvin")
    # parser.add_argument("--system", type=str, default="lips", help="System name")

    # args = parser.parse_args()
    # main(args)
    main()