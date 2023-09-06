import argparse
import numpy as np
import matplotlib.pyplot as plt
import ase
from ase.io import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory
from ase.geometry.analysis import Analysis
import seaborn as sns, numpy as np
import pandas as pd

#Example command:

"""
python angle_dist.py \
--traj_path "/home/civil/btech/ce1180169/MDBENCHGNN/example_mdbenchgnn/lips20_small/data/test/botnet_g80.xyz" \
--ref_traj_path "/home/civil/btech/ce1180169/MDBENCHGNN/example_mdbenchgnn/lips20_small/data/test/botnet_g75.xyz" \
--out_path  "/home/civil/btech/ce1180169/MDBENCHGNN/example_mdbenchgnn/lips20_small/data/test/" \
--angle 'P-S-P' \
--sys_name 'lips20_g80' 

"""


def main(args=None):
    Traj = ase.io.read(args.traj_path, index=':', format='extxyz')
    Traj_ref=read(args.ref_traj_path, index=':', format='extxyz')
    
    Angle_elems=args.angle
    A,B,C=Angle_elems.split('-')
    analysis = Analysis(Traj)
    analysis_ref = Analysis(Traj_ref)
    
    print("Calculating angles....")
    Angles=analysis.get_values(analysis.get_angles(A=A,B=B,C= C, unique=True))
    Angles_mean=np.mean(np.array(Angles),axis=0)
    
    Angles_ref=analysis_ref.get_values(analysis_ref.get_angles(A=A,B=B,C= C, unique=True))
    Angles_mean=np.mean(np.array(Angles_ref),axis=0)
    
    
    sns.displot(Angles_mean,kind="kde")
    sns.displot(Angles_mean_ref,kind="kde")
    
    plt.xlabel(Angle_elems+' Angle')
    plt.savefig(args.out_path+'/'+args.sys_name+"_"+Angle_elems+".png") 
    
    data = pd.DataFrame({Angle_elems: Angles_mean})
    data.to_csv(args.out_path+"/"+args.sys_name+"_"+Angle_elems+".csv")
    print("Done..., check ouput at "+args.out_path)    
    
       
#plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and plot RDF")
    parser.add_argument("--traj_path", required=True, help="Path to trajectory file")
    parser.add_argument("--ref_traj_path", required=True, help="Path to  reference trajectory file")
    
    parser.add_argument("--out_path", required=True, help="Path to output file")
    parser.add_argument("--angle", default='Li-P-S', help="Angle to calculate in format 'Li-P-S' ")
    parser.add_argument("--sys_name", type=str, default='System', help="System name")
    args = parser.parse_args()
    
    main(args)
