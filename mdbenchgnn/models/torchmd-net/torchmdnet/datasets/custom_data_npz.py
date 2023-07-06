import torch
from torch_geometric.data import InMemoryDataset,  Data
import numpy as np
from torch.utils.data import  Subset
import os


class CUSTOMDATASET(InMemoryDataset):
    """adapted from md17 
    """


    def __init__(self, root, energy_key, forces_key, positions_key, atomic_num_key, transform=None, pre_transform=None):

        self.energy_key = energy_key
        self.forces_key = forces_key
        self.positions_key = positions_key
        self.atomic_num_key = atomic_num_key
        super(CUSTOMDATASET, self).__init__(root, transform, pre_transform)# it runs the process function


        self.offsets = [0]
        self.data_all, self.slices_all = [], []
        for path in self.processed_paths:# ['/home/sire/phd/srz228573/equiformer/data_sl/equiformer_data/md17/aspirin/processed/md17-aspirin.pt']
            data, slices = torch.load(path)#
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
        return super(CUSTOMDATASET, self).get(idx - self.offsets[data_idx])

    @property
    def raw_file_names(self):
        return ["nequip_npz.npz"]#[MD17.molecule_files[mol] for mol in self.molecules]# this name of npz file to be loaded

    @property
    def processed_file_names(self):
        return ["md17.pt"]# ['/home/sire/phd/srz228573/equiformer/data_sl/equiformer_data/md17/aspirin/processed/md17-aspirin.pt']

    # def download(self):
    #     for file_name in self.raw_file_names:
    #         download_url(MD17.raw_url + file_name, self.raw_dir)

    def process(self):
        path = self.root + "/nequip_npz.npz"
        #print("----------",self.atomic_num_key, self.positions_key, self.energy_key, self.forces_key)
        # for path in self.raw_paths:# output of function raw_file_names
        data_npz = np.load(path)#/home/sire/phd/srz228573/equiformer/data_sl/equiformer_data/md17/aspirin/raw/md17_aspirin.npz
        z = torch.from_numpy(data_npz[self.atomic_num_key]).long()
        positions = torch.from_numpy(data_npz[self.positions_key]).float()
        energies = torch.from_numpy(data_npz[self.energy_key]).float()
        forces = torch.from_numpy(data_npz[self.forces_key]).float()
        
        samples = []
        for pos, y, neg_dy in zip(positions, energies, forces):# 1000 times, pos(83,3)
            samples.append(Data(z=z, pos=pos, y=y.unsqueeze(1), neg_dy=neg_dy))#

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]

        data, slices = self.collate(samples)
        torch.save((data, slices), self.root + "/processed/md17.pt")# ceates processed files


def get_datasets(root, energy_key, forces_key, positions_key, atomic_num_key):
    '''
        Return training, validation and testing sets of MD17 with the same data partition as TorchMD-NET.
    '''

    all_dataset = CUSTOMDATASET(root, energy_key, forces_key, positions_key, atomic_num_key)
    
    # hack for using subset class as in original
    subset_indices = list(range(len(all_dataset)))
    subset = Subset(all_dataset, subset_indices)
    train_dataset = subset

    return train_dataset


        