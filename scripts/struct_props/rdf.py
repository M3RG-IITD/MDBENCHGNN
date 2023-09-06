import ase
from ase import Atoms
from ase.geometry.analysis import Analysis
import numpy as np
from ase.io import read
import argparse
import matplotlib.pyplot as plt
import pandas as pd

#Example command:

"""
python rdf.py \
--traj_path "/home/civil/btech/ce1180169/MDBENCHGNN/example_mdbenchgnn/lips20_small/data/test/botnet_g80.xyz" \
--ref_traj_path "/home/civil/btech/ce1180169/MDBENCHGNN/example_mdbenchgnn/lips20_small/data/test/botnet_g80.xyz" \
--out_path  "/home/civil/btech/ce1180169/MDBENCHGNN/example_mdbenchgnn/lips20_small/data/test/" \
--sys_name 'lips20_g80' 

"""


def main(args=None):
    
    Traj = read(args.traj_path, index=':', format='extxyz')
    Traj_ref=read(args.ref_traj_path, index=':', format='extxyz')
    
    rmin = args.rmin
    dr = args.dr
    rmax = np.round(np.min([(np.min(Traj[0].cell.cellpar()[:3]) - 1) / 2, args.rmax]), 1)
    nbins = int((rmax - rmin) / dr)
    r = np.linspace(rmin, rmax, nbins)

    analysis = Analysis(Traj)
    analysis_ref = Analysis(Traj_ref)
    
    print("Calculating RDF .....")
    if(args.elements!=None):
        elements = (args.elements).split('-')
        rdf = analysis.get_rdf(rmax=rmax, nbins=nbins, imageIdx=None, elements=elements, return_dists=False)
        rdf_ref = analysis_ref.get_rdf(rmax=rmax, nbins=nbins, imageIdx=None, elements=elements, return_dists=False)    
        
        yaxis_label='Partial RDF: '+args.elements
    else:
        rdf = analysis.get_rdf(rmax=rmax, nbins=nbins, imageIdx=None, elements=None, return_dists=False)
        rdf_ref = analysis_ref.get_rdf(rmax=rmax, nbins=nbins, imageIdx=None, elements=None, return_dists=False)
        
        yaxis_label='Total RDF'
    g_r = np.mean(np.array(rdf), axis=0)
    g_r_ref = np.mean(np.array(rdf_ref), axis=0)


    plt.plot(r, g_r)
    plt.plot(r, g_r_ref,linestyle='--')
    plt.xlabel('Distance (Angstrom)')
    plt.ylabel(yaxis_label)
    plt.title('Radial Distribution Function')
    plt.legend(['Generated PDF','Reference PDF'])
    
    plt.savefig(args.out_path+"/"+args.sys_name+"_"+yaxis_label+".png")
    
    data = pd.DataFrame({'r': r, 'g_r': g_r, 'g_r_ref': g_r_ref })
    data.to_csv(args.out_path+"/"+args.sys_name+"_"+yaxis_label+".csv")
    print("Done..., check ouput at "+args.out_path)    
    #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and plot RDF")
    parser.add_argument("--traj_path", required=True, help="Path to trajectory file")
    parser.add_argument("--ref_traj_path", required=True, help="Path to  reference trajectory file")
    parser.add_argument("--out_path", required=True, help="Path to output file")
    parser.add_argument("--rmin", type=float, default=0.0, help="Minimum distance for RDF (Angstrom)")
    parser.add_argument("--rmax", type=float, default=10.0, help="Maximum distance for RDF (Angstrom)")
    parser.add_argument("--dr", type=float, default=0.05, help="Bin size for histogram (Angstrom)")
    parser.add_argument("--elements", type=str, default=None, help="Elements for partial pdf in format 'A-B' ")
    parser.add_argument("--sys_name", type=str, default='System', help="System name")
    
    args = parser.parse_args()
    
    main(args)
    

