import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase.io import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory
import seaborn as sns, numpy as np

def main(args=None):
    Traj = ase.io.read(args.traj_path, index=':', format='extxyz')

    Angle_elems=args.angle
    A,B,C=Angle_elems.split('-')
    analysis = Analysis(Traj)
    Angles=analysis.get_values(analysis.get_angles(A=A,B=B,C= C, unique=True))
    Angles_mean=np.mean(np.array(Angles),axis=0)
    sns.displot(Angles_mean,kind="kde")
    plt.xlabel(Angle_elems+' Angle')
    plt.savefig(args.outpath+'/'+Angle_elems+".png")    
#plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and plot RDF")
    parser.add_argument("--traj_path", required=True, help="Path to trajectory file")
    parser.add_argument("--out_path", required=True, help="Path to output file")
    parser.add_argument("--angle", default='Li-P-S', help="Angle to calculate in format 'Li-P-S' ")
    
    
    main(args)
