Data formats used in this package:
I) The .npz format is a file format used by NumPy. It can store multiple arrays within a single .npz file, making it convenient for bundling related data together. Each array in the .npz file is associated with a unique key, which can be used to access the specific array when loading the data.

II) The Extended XYZ (.xyz) format is a file format commonly used for representing molecular structures and atomic coordinates. It provides a simple and flexible way to store atomic positions, element types, and additional properties associated with each atom. Each frame can represent a different time step, molecular conformation, or related structure.

NPZ and XYZ can be interconverted b/w each other.




Information needed:

1. 'atomic_numbers'
2. 'cell' [required for mace and botnet (eg: MD17 data doesnt have cell info at this point)]
3. 'forces'
4. 'energy'
5. 'positions'
6. 'pbc' #preriodic boundary condition in all 3 directions

example of accepted data: present at example/lips/data
    =============Lips data info============
    Keys for data:
    key: pbc with shape (3,)
    key: pos with shape (1000, 83, 3)
    key: energy with shape (1000, 1)
    key: forces with shape (1000, 83, 3)
    key: cell with shape (3, 3)
    key: atomic_numbers with shape (83,)

NOTE: 
1. Nequip, Allegro, Equiformer and TorchmdNET can work with just atomic numbers, forces, energy and positions.
2. name the xyz file as botnet.xyz and npz as nequip_npz.npz to avoid error's


Mace and Botnet : input file -> xyz format
Nequip, Allegro, Equiformer and torchmdNET : input file -> npz format


converting XYZ to NPZ:
run file xyz_to_npz


CONVERTING NPZ to XYZ:
example: 
python preprocessing/npz_to_xyz.py --input-filepath example/lips/data/val/nequip_npz.npz
will create an xyz file at example/lips/data/val/botnet.xyz


IMP NOTE:
In case facing some error in data conversion: 
1) get data in npz format and cross check with example/lips/data/train/nequip_npz.npz
2) Then create test, train and val split like we have here example/lips/data
3) Then convert npz to xyz 

