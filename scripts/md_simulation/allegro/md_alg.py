from nequip.ase import NequIPCalculator
import torch
from ase import units
from ase.md.langevin import Langevin
#from ase.md.nvtberendsen import NVTBerendsen
from ase.io import read, write
from tqdm import tqdm

def main():
    """Run MD simulation with NequIP model
    """
    model_path = "example/lips/allegro/deployed.pth"
    calculator = NequIPCalculator.from_deployed_model(model_path= model_path, device='cuda')
    device = "cuda"; model_name = "allegro"
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