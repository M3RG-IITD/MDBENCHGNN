import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from pytorch_lightning.utilities import rank_zero_warn
import numpy as np
import os
from torch.utils.data import Dataset, Subset


class MD17(InMemoryDataset):
    """Machine learning of accurate energy-conserving molecular force fields (Chmiela et al. 2017)
    This class provides functionality for loading MD trajectories from the original dataset, not the revised versions.
    See http://www.quantum-machine.org/gdml/#datasets for details.
    """

    raw_url = "http://www.quantum-machine.org/gdml/data/npz/"

    # molecule_files = dict(
    #     aspirin="aspirin_dft.npz",
    #     benzene="benzene_old_dft.npz",
    #     ethanol="ethanol_dft.npz",
    #     malonaldehyde="malonaldehyde_dft.npz",
    #     naphthalene="naphthalene_dft.npz",
    #     salicylic_acid="salicylic_dft.npz",
    #     toluene="toluene_dft.npz",
    #     uracil="uracil_dft.npz",
    # )
    molecule_files = dict(
        aspirin="md17_aspirin.npz",
        benzene="md17_benzene2017.npz",
        ethanol="md17_ethanol.npz",
        malonaldehyde="md17_malonaldehyde.npz",
        naphthalene="md17_naphthalene.npz",
        salicylic_acid="md17_salicylic.npz",
        toluene="md17_toluene.npz",
        uracil="md17_uracil.npz",
    )
    

    available_molecules = list(molecule_files.keys())

    def __init__(self, root, transform=None, pre_transform=None, molecules=None):
        # assert molecules is not None, (
        #     "Please provide the desired comma separated molecule(s) through"
        #     f"'molecules'. Available molecules are {', '.join(MD17.available_molecules)} "
        #     "or 'all' to train on the combined dataset."
        # )

        # if molecules == "all":
        #     molecules = ",".join(MD17.available_molecules)
        # self.molecules = molecules.split(",")

        # for mol in self.molecules:
        #     if mol not in MD17.available_molecules:
        #         raise RuntimeError(f"Molecule '{mol}' does not exist in MD17")

        # if len(self.molecules) > 1:
        #     rank_zero_warn(
        #         "MD17 molecules have different reference energies, "
        #         "which is not accounted for during training."
        #     )

        super(MD17, self).__init__(root, transform, pre_transform)

        self.offsets = [0]
        self.data_all, self.slices_all = [], []
        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(
                len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1]
            )
            # Extract the directory path
            directory_path = os.path.dirname(path)
            import shutil
            shutil.rmtree(directory_path) # remove processed folder as it is not needed anymore

    def len(self):
        return sum(
            len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all
        )

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(MD17, self).get(idx - self.offsets[data_idx])

    @property
    def raw_file_names(self):#A list of files in the raw_dir which needs to be found in order to skip the download.
        return ["nequip_npz.npz"]#[MD17.molecule_files[mol] for mol in self.molecules]# this name of npz file to be loaded

    @property
    def processed_file_names(self):# A list of files in the processed_dir which needs to be found in order to skip the processing.
        return ["md17.pt"]# ['/home/sire/phd/srz228573/equiformer/data_sl/equiformer_data/md17/aspirin/processed/md17-aspirin.pt']

    # def download(self):#Downloads raw data into raw_dir
    #     for file_name in self.raw_file_names:
    #         download_url(MD17.raw_url + file_name, self.raw_dir)

    def process(self):#Processes raw data and saves it into the processed_dir
        raw_path = self.root + "/nequip_npz.npz"
        #for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
        data_npz = np.load(raw_path)#'/home/sire/phd/srz228573/torchmd-net/bench_data_sl/torchmdnet_data/md17/aspirin/raw/md17_aspirin.npz'
        z = torch.from_numpy(data_npz["atomic_numbers"]).long()
        positions = torch.from_numpy(data_npz['pos']).float()
        energies = torch.from_numpy(data_npz['energy']).float()
        forces = torch.from_numpy(data_npz["forces"]).float()

        samples = []
        for pos, y, neg_dy in zip(positions, energies, forces):
            samples.append(Data(z=z, pos=pos, y=y.unsqueeze(1), neg_dy=neg_dy))

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]

        data, slices = self.collate(samples)
        torch.save((data, slices), self.root + "/processed/md17.pt")# ceates processed files


def get_md17_datasets(root):
    '''
        Return training, validation and testing sets of MD17 with the same data partition as TorchMD-NET.
    '''

    all_dataset = MD17(root)
    
    # hack for using subset class as in original
    subset_indices = list(range(len(all_dataset)))
    subset = Subset(all_dataset, subset_indices)
    train_dataset = subset

    return train_dataset
