"""Short script to generate a list of subhaloes to process."""

import argparse
import numpy as np
import hydrangea as hy
import os

from pdb import set_trace

def main():
    parser = argparse.ArgumentParser(
        description="Prepare simulation data for SKIRT processing.")
    parser.add_argument(
        '-c', '--simulation', type=int, help="Simulation index", default=0)
    parser.add_argument(
        '-x', '--snapshot', type=int, help="Snapshot index", default=29)
    parser.add_argument(
    	'-m', '--mstar_range', type=float, nargs='*', default=[0, 1e20],
    	help="Stellar mass range [M_Sun]"
    )
    parser.add_argument(
    	'-d', '--dist_from_cluster', type=float,
    	help="Max distance from the most massive halo in the simulation, "
    	     "in units of that halo's R200."
    )
    parser.add_argument(
    	'-s', '--ssfr_range', type=float, nargs='*', default=[0, 1e20],
    	help="Specific star formation rate range [yr^-1]"
    )
    parser.add_argument('-f', '--file', default='subhalo_list.txt',
    	help="Output file to write to (default: subhalo_list.txt)"
    )
    args = parser.parse_args()
    
    sim = hy.Simulation(args.simulation)
    sub = hy.SplitFile(sim.get_subfind_file(args.snapshot), 'Subhalo')

    sub.ssfr = sub.StarFormationRate / sub.MassType[:, 4]
    bflag = hy.hdf5.read_data(
    	sim.sh_extra_loc, f"Snapshot_{args.snapshot:03d}/BoundaryFlag")

    ssfr_range = args.ssfr_range
    mstar_range = args.mstar_range
    ind_sel = np.nonzero((sub.MassType[:, 4] >= mstar_range[0]) &
                         (sub.MassType[:, 4] < mstar_range[1]) &
                         (sub.ssfr >= ssfr_range[0]) &
                         (sub.ssfr < ssfr_range[1]) &
                         (bflag < 2) &
                         (sub.SubGroupNumber*0 == 0)
                        )[0]

    if args.dist_from_cluster is not None:
    	pos_sel = sub.CentreOfPotential[ind_sel, :]
    	fof_cl = hy.SplitFile(sim.get_subfind_file(args.snapshot), 'FOF',
    		                  read_index=0)
    	pos_cl = fof_cl.GroupCentreOfPotential
    	r_sel = np.linalg.norm(pos_sel - pos_cl, axis=1)
    	r_sel /= fof_cl.Group_R_Crit200
    	ind_subsel = np.nonzero(r_sel <= args.dist_from_cluster)[0]
    	ind_sel = ind_sel[ind_subsel]

    print(f'There are {len(ind_sel)} galaxies satisfying the criteria.')
    header=(
    	f"Subhaloes in simulation CE-{args.simulation}, "
    	f"snapshot {args.snapshot} (N = {len(ind_sel)}).\n"
	    f"M_star: {mstar_range[0]:.3e} - {mstar_range[1]:.3e} M_Sun,\n"
	    f"sSFR: {ssfr_range[0]:.3e} - {ssfr_range[1]:.3e} yr^-1.\n"
	    f"Max distance from central cluster: {args.dist_from_cluster} R_200."
    )

    # Create output directory if needed
    outdir = os.path.dirname(args.file)
    if not outdir == '' and not os.path.isdir(outdir):
        os.makedirs(outdir)

    np.savetxt(args.file, ind_sel, fmt='%d', header=header)

if __name__ == "__main__":
    main()
