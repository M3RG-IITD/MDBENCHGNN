from mdbenchgnn.utils.tmdnet_ase.tmdnet_calc import TmdnetCalculator
import torch
from ase import units
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.io import read, write
from tqdm import tqdm
import sys
sys.path.append("mdbenchgnn/models/torchmd-net/torchmdnet/models")

from model import load_model


def main():
    device = "cuda"; model_name = "tmdnet"
    model = load_model("example/lips/torchmdnet/epoch=129-val_loss=0.0184-test_loss=0.0941.ckpt", derivative = True)
    # see log for r_max : radius cutoff
    calculator = TmdnetCalculator(model=model, device='cpu', r_max = 5)
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