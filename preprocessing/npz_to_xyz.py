import numpy as np

from ase import Atoms
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-filepath', type=str, help='path to npz file') 
    #parser.add_argument('--output-filepath', help='path to npz file') 
    args = parser.parse_args()
    return args
    

def main():
    args = get_args()
    in_filename = args.input_filepath
    directory_path = "/".join(in_filename.split("/")[:-1]) + "/"
    out_filename = directory_path + "botnet.xyz"
    data = np.load(in_filename)
    
    print("keys for data:")
    for keys in data.files:
        print("key:",keys)
    
    print("\n")
    print("Please set above keys appropriately in script to avoid error")
    print("\n")
        
    positions = data['pos']
    cells = data['cell']
    atomic_numbers = data["atomic_numbers"]
    energies = data['energy']
    forces = data['forces']
    pbc_val = data["pbc"][0]# along one axis works
    print("\n")
    print(f"note if there is already a file at{out_filename}, this will append to it. Delete it beforehand if appending should be avoided")
    # iterate over data and write continuously to extxyz file
    for idx in range(len(positions)):# number of time steps
        curr_atoms = Atoms(
            # set atomic positions
            positions=positions[idx],
            # set cell in case it exists
            #cell=cells[idx],
            cell = cells,
            numbers=atomic_numbers,
            # set chemical symbols / species
            #symbols=atomic_numbers[idx], 
            # assuming data with periodic boundary conditions, set to false for e.g. for molecules in vacuum
            pbc=pbc_val # True or false
        )
        # set calculator to assign targets
        calculator = SinglePointCalculator(curr_atoms, energy=energies[idx], forces=forces[idx])
        curr_atoms.calc = calculator
        write(out_filename, curr_atoms, format='extxyz', append=True)
    
    print("\n")
    print("file created at", out_filename)

if __name__ == "__main__":
    main()
 
