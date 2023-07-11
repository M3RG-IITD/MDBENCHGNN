from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write
import numpy as np
import time
import torch
import sys
sys.path.append("mdbenchgnn/models/mace")
from mace.calculators import MACECalculator
from tqdm import tqdm
torch.set_default_dtype(torch.float64)

def main():
    calculator = MACECalculator(model_path='example/lips/mace/MACE_model_500_lips_swa.model', device='cuda', default_dtype='float64')
    calculator.model.double() # change model weights type to double precision(hack to avoid error)
    device = "cuda"; model_name = "mace"
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
