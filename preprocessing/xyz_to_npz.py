import numpy as np
from ase.io import read
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-filepath', type=str, help='path to xyz file') 
    #parser.add_argument('--output-filepath', help='path to npz file') 
    args = parser.parse_args()
    return args
    

def main():
    args = get_args()
    in_filename = args.input_filepath
    directory_path = "/".join(in_filename.split("/")[:-1]) + "/"
    out_filename = directory_path + "nequip_npz.npz"
    
    
    
    atoms = read(in_filename, index=':', format='extxyz')
    n_points = len(atoms)
    positions, cell, atomic_numbers, energy, forces = [], [], [], [], []
    for i in range(n_points):
        positions.append(atoms[i].get_positions())
        cell.append(atoms[i].get_cell())
        atomic_numbers.append(atoms[i].get_atomic_numbers())
        energy.append(atoms[i].get_potential_energy())
        forces.append(atoms[i].get_forces())
    positions = np.array(positions)
    cell = np.array(cell)[0]
    atomic_numbers = np.array(atomic_numbers)[0]
    energy = np.array(energy)[:, None] 
    forces = np.array(forces)
    
    data = {}
    data['pbc'] = np.array([True]*3)# preriodic boundary condition in all 3 directions
    data['pos'] = positions#[ranges[spidx]]
    data['energy'] = energy#[ranges[spidx]]
    data['forces'] = forces#[ranges[spidx]]
    data['cell'] = cell
    data['atomic_numbers'] = atomic_numbers
    np.savez(out_filename, **data)
    print("\n")
    print("file created at", out_filename)
    
if __name__ == "__main__":
    main()