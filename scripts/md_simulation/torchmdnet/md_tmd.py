import torch
from ase import units
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.io import read, write
from tqdm import tqdm
import sys
sys.path.append("/home/civil/btech/ce1180169/MDBENCHGNN")
sys.path.append("mdbenchgnn/models/torchmd-net/torchmdnet/models")
from mdbenchgnn.utils.tmdnet_ase.tmdnet_calc import TmdnetCalculator
from model import load_model



def main(args):
    model_name = "tmdnet"
    model = load_model(args.model_path, derivative = True)
    # see log for r_max : radius cutoff
    calculator = TmdnetCalculator(model=model, device=args.device, r_max = args.r_max)
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
    parser = argparse.ArgumentParser(description="Run MD simulation with TorchMD-NET model")
    parser.add_argument("--model_path", type=str, default="Output_mdbgnn/tmdnet/md17/aspirin/epoch=2739-val_loss=0.1136-test_loss=0.1902.ckpt", help="Path to the model checkpoint")
    parser.add_argument("--init_conf_path", type=str, default="example/lips20/data/test/botnet.xyz", help="Path to the initial configuration")
    parser.add_argument("--device", type=str, default="cpu", help="Device:["cpu","cuda"]")
    parser.add_argument("--init_conf_N", type=int, default=10, help="No. of initial configuration, [i=0 to N-1] confs read")
    parser.add_argument("--out_dir", type=str, default="out_dir_sl/neqip/lips20/", help="Output path")
    parser.add_argument("--temp", type=float, default=520, help="Temperature in Kelvin")
    parser.add_argument("--timestep", type=float, default=1.0, help="Timestep in fs units")
    parser.add_argument("--runsteps", type=int, default=1000, help="No. of steps in to run")
    parser.add_argument("--r_max", type=float, default=5.0, help="r_max")
    
    
    args = parser.parse_args()
    main(args)
