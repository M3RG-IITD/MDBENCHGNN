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
    
    print("=============Lips data info============")
    data = {
    'pbc': (3,),
    'pos': (1000, 83, 3),
    'energy': (1000, 1),
    'forces': (1000, 83, 3),
    'cell': (3, 3),
    'atomic_numbers': (83,)
    }

    print("Keys for data:")
    for key, shape in data.items():
        print(f"key: {key} with shape {shape}")
    
    print("\n")
    print("=============your input data info============")
    args = get_args()
    in_filename = args.input_filepath
    data = np.load(in_filename)
    
    
    print("keys for data:")
    for keys in data.files:
        print(f"key:{keys} with shape {data[keys].shape}")
    

if __name__ == "__main__":
    main()
 
