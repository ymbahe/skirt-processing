# Small helper script to extract data for one galaxy from Hydrangea
# for SKIRT imaging.

import numpy as np
import argparse
import hydrangea as hy
from pdb import set_trace
import image_routines as ir
from scipy.spatial import cKDTree
import os

stars_file = 'sim_stars.txt'
dust_file = 'sim_dust.txt'
hii_file = 'sim_hii.txt'

T_max = 8000   # Max temperature of dust-bearing non-SF gas [K]
f_dust = 0.3   # Dust-to-metal mass ratio [-]
min_age = 0.1  # Minimum age of "plain" stars [Gyr]

def main(ind_sub=None):
    """
    Wrapper to run the particle extraction as a script.

    Parameters
    ----------
    isub : int, optional
        The subhalo to process. If None (default), find one based on
        hard-coded properties...
    """
    global kernel

    parser = argparse.ArgumentParser(
        description="Prepare simulation data for SKIRT processing.")
    parser.add_argument(
        '-s', '--subhalo', type=int, help="The subhalo index to process")
    parser.add_argument(
        '-c', '--simulation', type=int, help="Simulation index", default=0)
    parser.add_argument(
        '-x', '--snapshot', type=int, help="Snapshot index", default=29)
    parser.add_argument(
        '-d', '--directory', help="Directory in which to store output",
        default=".")
    args = parser.parse_args()
    if not os.path.isdir(args.directory):
        os.makedirs(args.directory)

    isim = args.simulation
    isnap = args.snapshot
    ind_sub = args.subhalo

    if ind_sub is None:
        ind_sub = find_subhalo_list(
            isim, isnap, mstar_range=(3e10, 5e11), ssfr_range=(1e-10, 1e-9))[0]
    print(f"Processing subhalo {ind_sub} in sim {isim}, snapshot {isnap}.")

    kernel = TruncGauss()

    galaxy = Galaxy(isim, isnap, ind_sub)

    # Initialize star sources: only old ones
    stars = StarSource(galaxy)

    # Initialize dust sources: only non-star forming gas
    dust = DustMedium(galaxy)

    # Load and resample SF regions
    sfregions = SFRegions(galaxy)
    sfregions.resample_particles()

    # Build and write catalogue of HII regions from re-sampled SF regions
    hii = SFRSource(sfregions.sub_props)
    hii.write_data(isim, ind_sub, isnap, args.directory)

    # Add re-sampled young stars to catalogue and write
    stars.incorporate_resampled_stars(sfregions.sub_props)
    stars.write_data(isim, ind_sub, isnap, args.directory)

    # For dust, we need to add two extra populations:
    # (i): re-sampled `passive' SF regions...
    dust.incorporate_resampled_gas(sfregions.sub_props)

    # (ii): ... and `ghost' particles to account for dust within HII regions
    dust.incorporate_hii_ghosts(hii)

    dust.write_data(isim, ind_sub, isnap, args.directory)


def find_subhalo_list(isim, isnap, mstar_range, ssfr_range):
    """Find subhaloes in a specified mstar and ssfr range."""
    sim = hy.Simulation(isim)
    sub = hy.SplitFile(sim.get_subfind_file(isnap), 'Subhalo')
    sub.ssfr = sub.StarFormationRate / sub.MassType[:, 4]
    ind_sel = np.nonzero((sub.MassType[:, 4] >= mstar_range[0]) &
                         (sub.MassType[:, 4] < mstar_range[1]) &
                         (sub.ssfr >= ssfr_range[0]) &
                         (sub.ssfr < ssfr_range[1]) &
                         (sub.SubGroupNumber == 0)
                        )[0]
    print(f'There are {len(ind_sel)} galaxies satisfying the criteria.')
    return ind_sel


class TruncGauss:
    """
    Class to handle sampling within a truncated Gaussian kernel.

    See SKIRT-9 documentation for details.
    """
    def __init__(self):
        """Set up interpolation."""
        self.N = 2.56810060330949540082
        self.A = -5.85836755024609305208
        self.Nu = 401

        self.xvec = np.zeros(self.Nu)
        self.du = 1./(self.Nu-1)
        self.uvec = np.linspace(0, 1, self.Nu)

        u_mid = 0.5 * (self.uvec[:-1] + self.uvec[1:])
        mid_weights = self.weights(u_mid)

        for ibin in range(self.Nu-1):
            curr_weight = mid_weights[ibin]
            curr_u = self.uvec[ibin]
            self.xvec[ibin+1] = (
                self.xvec[ibin] + 4*np.pi*curr_weight * curr_u**2 * self.du)
        self.xvec[-1] = 1.0

    def weights(self, u):
        """Calculate the kernel weight function for normalized radii u."""
        weights = np.zeros_like(u)
        ind_good = np.nonzero(u < 1)[0]
        weights[ind_good] = self.N * np.exp(self.A * u**2)

        return weights

    def sampled_radii(self, num):
        """Returns a specified number of sampled radii from the kernel."""
        xrand = np.random.random(num)
        ind_low = np.searchsorted(self.xvec, xrand) - 1

        f_interpol = ((xrand - self.xvec[ind_low])
                      /(self.xvec[ind_low+1] - self.xvec[ind_low]))
        u = self.uvec[ind_low] + self.du * f_interpol
        return u

    def sampled_offsets(self, num, hsml):
        """
        Returns a specified number of 3D spatial offsets from the centre.

        Parameters:
        -----------
        num : int
            The number of points to generate
        hsml : float
            The smoothing radius of the kernel

        Returns:
        --------
        offsets : ndarray(num, 3)
            The sampled offsets from the kernel centre, in the same units
            as hsml.
        """
        r = self.sampled_radii(num) * hsml
        phi = np.random.random(num) * 2*np.pi
        theta = np.arccos(1 - 2 * np.random.random(num))

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        offsets = np.vstack((x, y, z)).T

        if np.count_nonzero(np.ravel(offsets) * 0 != 0) > 0:
            print(f'Nan warning...')
            set_trace()

        return offsets


class Galaxy:
    """Representation of global galaxy properties.

    Parameters
    ----------
    isim : int
        Index of the simulation from which the galaxy is taken.
    isnap : int
        Snapshot at which the galaxy is observed.
    isub : int
        Subhalo index of the galaxy.
    """

    def __init__(self, isim, isnap, isub):
        sim = hy.Simulation(isim)
        snap_file = sim.get_snapshot_file(isnap)
        sub_file = sim.get_subfind_file(isnap)

        self.subhalo = hy.SplitFile(sub_file, 'Subhalo', read_index=isub)
        self.centre = self.subhalo.CentreOfPotential

        # Set up targetted reader for gas and star particles
        self.gas = hy.ReadRegion(snap_file, 0, self.centre, 0.05, exact=True)
        self.stars = hy.ReadRegion(snap_file, 4, self.centre, 0.05, exact=True)

        # Some specific post-processing steps
        self.get_relative_coordinates()
        self.get_stellar_hsml()
        self.get_stellar_ages()

        self.gas.entropy_cgs = self.gas.read_data('Entropy', units='cgs')
        self.gas.density_cgs = self.gas.read_data('Density', units='cgs')

    def get_relative_coordinates(self):
        """
        Compute the offsets of star and gas particles from galaxy centre,
        in kpc.
        """
        self.stars.coordinates = (self.stars.Coordinates - self.centre) * 1e3
        self.gas.coordinates = (self.gas.Coordinates - self.centre) * 1e3

        # Also convert gas smoothing lengths to kpc, for consistency with stars
        self.gas.hsml = self.gas.SmoothingLength * 1e3

    def get_stellar_hsml(self, nngb=64):
        """(Re-)compute the stellar smoothing lengths.

        As in Trayford+17, the smoothing length is taken as the distance to
        the 64th nearest neighbouring star particle.
        """
        coordinates = self.stars.coordinates
        tree = cKDTree(coordinates)
        ngb_dists, ngb_lists = tree.query(coordinates, nngb, workers=-1)
        self.stars.hsml = ngb_dists[:, -1]

    def get_stellar_ages(self):
        """Compute age of star particles in Gyr."""
        self.stars.t_form = hy.aexp_to_time(self.stars.StellarFormationTime)
        self.stars.age = self.subhalo.time - self.stars.t_form


class StarSource:
    """Collection of (sub-/)particles treated as stellar sources.

    This includes both 'real' old star particles, and re-sampled sub-particles
    that are old enough to be treated as stars.
    """ 

    def __init__(self, galaxy):
        """Initialize the collection with old stars."""
        ind_oldstars = np.nonzero(galaxy.stars.age >= min_age)[0]

        self.coordinates = galaxy.stars.coordinates[ind_oldstars, :]
        self.hsml = galaxy.stars.hsml[ind_oldstars]
        self.masses = galaxy.stars.Mass[ind_oldstars]
        self.metallicities = galaxy.stars.SmoothedMetallicity[ind_oldstars]
        self.ages = galaxy.stars.age[ind_oldstars]
        self.kinds = np.zeros(len(self.ages), dtype=int)

    def incorporate_resampled_stars(self, subparticles):
        """Incorporate sub-particles resampled as stars."""
        ind_stars = np.nonzero(
            (subparticles['ages'] > 0.01) &
            (subparticles['ages'] <= min_age))[0]
        self.coordinates = np.concatenate(
            (self.coordinates,
                subparticles['parent_coordinates'][ind_stars, :]))
        self.hsml = np.concatenate(
            (self.hsml, subparticles['hsml'][ind_stars]))
        self.masses = np.concatenate(
            (self.masses, subparticles['masses'][ind_stars]))
        self.metallicities = np.concatenate(
            (self.metallicities, subparticles['metallicities'][ind_stars]))
        self.ages = np.concatenate(
            (self.ages, subparticles['ages'][ind_stars]))
        self.kinds = np.concatenate(
            (self.kinds, np.zeros(len(ind_stars), dtype=int) + 1))

    def write_data(self, isim, isub, isnap, directory):
        """Write out data in SKIRT compatible format."""

        array = np.hstack(
            (self.coordinates,
             self.hsml[:, None],
             self.masses[:, None],
             self.metallicities[:, None],
             self.ages[:, None]
            ))

        stars_header = (f'Stellar sources for galaxy CE-{isim}/S-{isub}' '\n'
               'Column 1: position x (kpc)\n'
               'Column 2: position y (kpc)\n'
               'Column 3: position z (kpc)\n'
               'Column 4: smoothing length (kpc)\n'
               'Column 5: mass (Msun)\n'
               'Column 6: Metallicities ()\n'
               'Column 7: Ages (Gyr)\n'
            )
        file_name = stars_file.replace('.txt', f'.{isim}.{isnap}.{isub}.txt')
        outfile = f"{directory}/{file_name}"
        np.savetxt(outfile, array, header=stars_header,
                   fmt='%3.4f %3.4f %3.4f %3.4f %.4e %.4e %.4e')


class DustMedium:
    """Collection of (sub-/)particles treated as dust medium."""

    def __init__(self, galaxy):
        """Initialize the collection with non-star-forming gas particles."""
        ind_coolgas = np.nonzero((galaxy.gas.StarFormationRate <= 0) &
                                 (galaxy.gas.Temperature < T_max))[0]

        self.coordinates = galaxy.gas.coordinates[ind_coolgas, :]
        self.hsml = galaxy.gas.hsml[ind_coolgas]
        self.dustMass = (galaxy.gas.Mass[ind_coolgas]
                         * galaxy.gas.SmoothedMetallicity[ind_coolgas]
                         * f_dust)
        self.origin = np.zeros(len(ind_coolgas), dtype=int)

    def incorporate_resampled_gas(self, subparticles):
        """Incorporate particles resampled as 'not yet formed'"""

        ind_dust = np.nonzero(subparticles['ages'] < 0)[0]
        if len(ind_dust) == 0:
            return

        dust_parents = subparticles['parent_ids'][ind_dust]

        unique_parents, first_sub = np.unique(dust_parents, return_index=True)
        m_by_parent, edges = np.histogram(
            subparticles['parent_ids'][ind_dust],
            weights=subparticles['masses'][ind_dust],
            bins=np.arange(np.max(unique_parents)+2))
        self.coordinates = np.concatenate(
            (self.coordinates,
             subparticles['parent_coordinates'][ind_dust[first_sub], :]))
        self.hsml = np.concatenate(
            (self.hsml, subparticles['hsml'][ind_dust[first_sub]]))
    
        subDustMass = (m_by_parent[unique_parents] *
                       subparticles['metallicities'][ind_dust[first_sub]] *
                       f_dust)
        self.dustMass = np.concatenate((self.dustMass, subDustMass))
        self.origin = np.concatenate(
            (self.origin, np.zeros(len(ind_dust), dtype=int) + 1))

    def incorporate_hii_ghosts(self, hii):
        """Incorporate 'ghost particles' at the location of HII regions."""
        self.coordinates = np.concatenate((self.coordinates, hii.coordinates))
        self.hsml = np.concatenate((self.hsml, 3*hii.radii))

        hiiDustMass = (-10 * hii.masses * hii.metallicities * f_dust)
        self.dustMass = np.concatenate((self.dustMass, hiiDustMass))
        self.origin = np.concatenate(
            (self.origin, np.zeros(len(hii.radii), dtype=int) + 2))

    def write_data(self, isim, isub, isnap, directory):
        """Write out data in SKIRT-compatible format."""

        array = np.hstack(
            (self.coordinates,
             self.hsml[:, None],
             self.dustMass[:, None]
             )
            )
        dust_header = (f'Dust sources for galaxy CE-{isim}/S-{isub}' '\n'
        'Column 1: position x (kpc)\n'
        'Column 2: position y (kpc)\n'
        'Column 3: position z (kpc)\n'
        'Column 4: smoothing length (kpc)\n'
        'Column 5: mass (Msun)\n'
        )
        file_name = dust_file.replace('.txt', f'.{isim}.{isnap}.{isub}.txt')
        outfile = f"{directory}/{file_name}"
        np.savetxt(outfile, array, header=dust_header,
            fmt='%3.4f %3.4f %3.4f %3.4f %.4e')
  

class SFRSource:
    """
    Collection of sub-particles treated as HII regions.

    These are only sourced from the re-sampled particles. For details,
    see Section 2 of Trayford et al. (2017).
    """
    def __init__(self, subparticles):
        ind_hii = np.nonzero(
            (subparticles['ages'] >= 0) & (subparticles['ages'] < 0.01))[0]

        self.num_hii = len(ind_hii)
        self.parent_coordinates = subparticles['parent_coordinates'][ind_hii]
        
        # Assume SFR corresponding to using up all sub-particle mass
        # within 10 Myr (consistent with assumed HII region life time)
        self.masses = subparticles['masses'][ind_hii]
        self.sfrs = self.masses / 1e7
        self.metallicities = subparticles['metallicities'][ind_hii]
        self.pressures = subparticles['pressures'][ind_hii]

        # "Compactness" parameter (see Trayford+17 and Groves+08)
        self.compactnesses = (
            3/5 * np.log10(self.masses)
            + 2/5 * np.log10(self.pressures / 1.38065e-16)
            )

        # Calculate the extent of the HII regions. For this, we need to convert
        # the (parent) density from m_H / cm^3 to M_Sun / kpc^3 
        self.densities = (
            subparticles['densities'][ind_hii] * 1.6726e-27 / 1.989e30 *
            (3.0857e21**3))
        self.radii = (8 * 10 * self.masses / (np.pi * self.densities))**(1/3)

        # Assign a position around the parent particle. This is done assuming
        # a somewhat reduced parent kernel radius,
        # r = sqrt(r_parent^2 - self.radii^2). In the rare cases where the
        # HII radius is greater than the parent kernel, r is set to zero.
        self.parent_hsml = subparticles['hsml'][ind_hii]
        self.kernels = np.sqrt(self.parent_hsml**2 - self.radii**2)
        ind_bad = np.nonzero(self.radii > self.parent_hsml)[0]
        self.kernels[ind_bad] = 0
        self.coordinates = np.zeros_like(self.parent_coordinates)
        for ipart in range(self.num_hii):
            self.coordinates[ipart, :] = (
                self.parent_coordinates[ipart, :]
                + kernel.sampled_offsets(1, self.kernels[ipart]))

    def write_data(self, isim, isub, isnap, directory):
        """Write out the HII region data to a SKIRT-compatible file."""

        array = np.hstack(
            (self.coordinates,
              self.radii[:, None],
              self.sfrs[:, None],
              self.metallicities[:, None],
              self.compactnesses[:, None],
              self.pressures[:, None] * 0.1,     # convert cgs --> Pa
              np.zeros((len(self.radii), 1)) + 0.1,   # f_PDR == 0.1
              )
            )

        hii_header = (f'HII regions for galaxy CE-{isim}/S-{isub}' '\n'
               'Column 1: position x (kpc)\n'
               'Column 2: position y (kpc)\n'
               'Column 3: position z (kpc)\n'
               'Column 4: smoothing length (kpc)\n'
               'Column 5: Star formation rates (Msun/yr)\n'
               'Column 6: Metallicities ()\n'
               'Column 7: Compactnesses ()\n'
               'Column 8: Pressures (Pa)\n'
               'Column 9: f_PDR ()'
            )
        file_name = hii_file.replace('.txt', f'.{isim}.{isnap}.{isub}.txt')
        outfile = f"{directory}/{file_name}"
        np.savetxt(outfile, array, header=hii_header,
            fmt='%3.4f %3.4f %3.4f %3.4f %.4e %.4e %.4e %.4e %1.2f'
        )


class SFRegions:
    """
    Collection of star forming regions in the simulation.

    They are sourced from young stars and star forming gas particles,
    and are re-sampled into sub-particles.
    """

    def __init__(self, galaxy, from_stars=True, from_gas=True):
        """Load input properties of stars and gas."""

        if from_stars:
            ind_youngstars = np.nonzero(galaxy.stars.age < min_age)[0]
            self.coordinates = galaxy.stars.coordinates[ind_youngstars, :]
            self.hsml = galaxy.stars.hsml[ind_youngstars]
            self.masses = galaxy.stars.InitialMass[ind_youngstars]
            self.metallicities = (
                galaxy.stars.SmoothedMetallicity[ind_youngstars])

            # For stars, we only have the birth density recorded. From this, we
            # estimate a star formation rate and pressure based on the imposed
            # entropy floor.
            self.densities = galaxy.stars.BirthDensity[ind_youngstars]
            self.sfrs = self.sfr_from_density(self.densities) * self.masses
            self.pressures = self.pressure_from_density(self.densities)

            # Before adding SF gas, make a record which particles are sourced
            # from stars (type == 4).
            self.types = np.zeros(len(self.hsml), dtype=int) + 4
        else:
            self.coordinates = np.zeros((0, 3))
            self.hsml = np.zeros(0)
            self.masses = np.zeros(0)
            self.metallicities = np.zeros(0)
            self.densities = np.zeros(0)
            self.sfrs = np.zeros(0)
            self.pressures = np.zeros(0)
            self.types = np.zeros(0, dtype=int)

        if from_gas:
            ind_sfgas = np.nonzero(galaxy.gas.StarFormationRate > 0)[0]
            self.coordinates = np.concatenate(
                (self.coordinates, galaxy.gas.coordinates[ind_sfgas, :]))
            self.hsml = np.concatenate(
                (self.hsml, galaxy.gas.hsml[ind_sfgas]))
            self.masses = np.concatenate(
                (self.masses, galaxy.gas.Mass[ind_sfgas]))
            self.metallicities = np.concatenate(
                (self.metallicities, galaxy.gas.SmoothedMetallicity[ind_sfgas]))
            self.densities = np.concatenate(
                (self.densities, galaxy.gas.Density[ind_sfgas]))
            self.sfrs = np.concatenate(
                (self.sfrs, galaxy.gas.StarFormationRate[ind_sfgas]))
            
            gas_pressure = (galaxy.gas.entropy_cgs[ind_sfgas]
                            * galaxy.gas.density_cgs[ind_sfgas]**(5/3))
            self.pressures = np.concatenate((self.pressures, gas_pressure))
            self.types = np.concatenate(
                (self.types, np.zeros(len(ind_sfgas), dtype=int)))

        # Set parameters

        m_min = 700                # Min re-sampled mass [M_Sun]
        m_max = 1e6                # Max re-sampled mass [M_Sun]
        resampling_index = -1.8    # Power-law index for mass function

        self.m1 = m_min**(resampling_index + 1)
        self.dm = m_max**(resampling_index + 1) - self.m1
        self.exponent = 1. / (resampling_index + 1)

    def resample_particles(self):
        """Re-sample the particles."""

        # Estimate the number of sub-particles to be generated.
        num_est = 0 if len(self.hsml) == 0 else (
            int(len(self.hsml) * np.mean(self.masses) / 4e3))

        self.sub_props = {
            'coordinates': np.zeros((num_est, 3), dtype=float),
            'parent_coordinates': np.zeros((num_est, 3), dtype=float),
            'masses': np.zeros(num_est),
            'hsml': np.zeros(num_est),
            'sfrs': np.zeros(num_est),
            'metallicities': np.zeros(num_est),
            'pressures': np.zeros(num_est),
            'parent_ids': np.zeros(num_est, dtype=int),
            'parent_types': np.zeros(num_est, dtype=int),
            'ages': np.zeros(num_est),
            'densities': np.zeros(num_est),
        }

        self.num_sub = 0
        self.num_allocated_sub = num_est

        for ipart in range(len(self.hsml)):
            self.resample_one_particle(ipart)

        # Shrink arrays to actually generated number of particles
        self.shrink_subparticle_arrays()

    def resample_one_particle(self, ipart):
        """Resample the i-th particle."""
        mass = self.masses[ipart]
        sampled_mass = 0
        sub_masses = []

        while True:
            f_random = np.random.random(1)[0]
            sub_mass = (self.m1 + (self.dm * f_random))**self.exponent

            # We abort the loop if adding this sub-particle would exceed the
            # actual particle mass
            if sampled_mass + sub_mass > mass:
                break

            sub_masses.append(sub_mass)
            sampled_mass += sub_mass

        # Once we get here, sampling is finished. See how many we got.
        num_sub = len(sub_masses)

        # In general, we have sampled slightly too little mass. Scale up all
        # re-sampled masses to exactly conserve the input mass.
        sub_masses = np.array(sub_masses) * (mass / sampled_mass)

        if ipart % 100 == 0:
            print(f"Re-sampled particle {ipart} into {num_sub} sub-particles.")
            print(f"Mass range: {sub_masses.min():.1f} - "
                  f"{sub_masses.max():.1f} M_Sun")

        if self.num_sub + num_sub > self.num_allocated_sub:
            self.expand_subparticle_fields()

        # Append properties of sub-particles to dict entries
        inds = np.arange(self.num_sub, self.num_sub+num_sub)
        self.sub_props['parent_ids'][inds] = (
            np.zeros(num_sub, dtype=int) + ipart)
        self.sub_props['parent_types'][inds] = (
             np.zeros(num_sub, dtype=int) + self.types[ipart])
        self.sub_props['masses'][inds] = sub_masses

        # Assign a shifted position within the parent kernel. This is
        # currently *not* actually used!
        self.sub_props['coordinates'][inds, :] = (
             self.coordinates[ipart, :] +
             kernel.sampled_offsets(num_sub, self.hsml[ipart]))

        # Parent star formation rates are distributed proportional to mass
        self.sub_props['sfrs'][inds] = (
            sub_masses * self.sfrs[ipart] / self.masses[ipart])

        # Metallicity, pressure, density, smoothing length, and parent
        # coordinates are copied directly from the parent particles
        self.sub_props['metallicities'][inds] = (
             np.zeros(num_sub) + self.metallicities[ipart])
        self.sub_props['pressures'][inds] = (
            np.zeros(num_sub) + self.pressures[ipart])
        self.sub_props['densities'][inds] = (
            np.zeros(num_sub) + self.densities[ipart])
        self.sub_props['hsml'][inds] = (
            np.zeros(num_sub) + self.hsml[ipart])
        self.sub_props['parent_coordinates'][inds, :] = (
             np.repeat(
                (self.coordinates[ipart, :])[:, None], num_sub, axis=1)).T

        # Ages are assigned individually for each subparticle
        resampled_ages = self.assign_resampled_ages(
            (self.masses[ipart] / self.sfrs[ipart]) / 1e9, num_sub)
        self.sub_props['ages'][inds] = resampled_ages

        self.num_sub += num_sub


    def sfr_from_density(self, density):
        """
        Estimate the star formation rate from the gas density.

        This is done assuming that the particle is exactly on the imposed
        entropy floor. Although star forming particles may in principle lie
        slightly above this as well, this assumption is empirically verified
        on gas particles at z = 0. In this case, the equations for SFR
        and entropy floor from Schaye+15 correspond to

        SFR/mass = 6.04 x 10^-10 yr^-1 * (density [m_H/cm^-3])^(4/15)

        For densities below the entropy floor threshold (n_H = 0.1 cm^-3), or
        rather break to the constant-temperature floor, the Jeans limiting
        floor would significantly underestimate the SFR. We therefore apply
        an empirically determined correction factor of
        1 + 0.15 * log_10(n_thresh/n), which gives an approximately unbiased
        estimate of the SFR.

        Parameters
        ----------
        density : ndarray(float)
            The density of the particles, in units of m_H / cm^-3

        Returns
        -------
        sfr : ndarray(float)
            The estimated SFR *per unit mass*

        """
        sfr_eos = 6.04e-10 * density**(4/15)

        rho_thresh = 0.1 / 0.752
        ind_low = np.nonzero(density < rho_thresh)[0]
        sfr_eos[ind_low] *= (1 + 0.15 * np.log10(rho_thresh/density[ind_low]))

        return sfr_eos

    def pressure_from_density(self, density):
        """
        Estimate the pressure from the gas density.

        This is done assuming that the particle lies exactly on the imposed
        entropy floor.

        The normalisation pressure (at the threshold of the entropy floor)
        corresponds to 865.958 k_B [cgs units], with a value of Boltzmann's
        constant k_B = 1.38065e-16 cm^2 g s^-2 K^-1.

        Currently, no correction is applied for gas below the threshold of the
        Jeans limiting entropy floor.

        Parameters
        ----------
        density : ndarray(float)
            The density of the particles, in units of m_H / cm^-3

        Returns
        -------
        pressure : ndarray(float)
            The estimated pressure, in cgs units.
        """
        p_eos = 865.958 * (density / 0.133)**(4/3) * 1.38065e-16

        return p_eos

    def assign_resampled_ages(self, tau, number):
        """
        Assign ages to freshly resampled particles.

        Ages are uniformly sampled over an interval of length tau, which is
        the ratio of (real) particle mass and star formation rate. If
        tau < 100 Myr, the interval stretches from 0 to tau. Otherwise, it
        stretches from (100 Myr - tau) to (tau - 100 Myr). In other words,
        starting from tau = 0 (infinitely high SFR), ages first extend up to
        a maximum of 100 Myr. But by assumption re-sampled particles cannot
        be older than 100 Myr, so if tau is longer than this, it is interpreted
        as "still to form stars" and assigned a negative age.

        The actual value of negative ages is unimportant; all those particles
        are used (exclusively) as dust absorbers.

        Parameters:
        -----------
        tau : float
            The star formation time scale, M / SFR, of the parent particle.
        number : int
            The number of ages to sample.

        Returns:
        --------
        ages : ndarray(number)
            Array of sampled ages.
        """
        if tau < min_age:
            age_min = 0
            age_max = tau
        else:
            age_min = min_age - tau
            age_max = min_age

        ages = np.random.random(number) * (age_max - age_min) + age_min
        return ages

    def expand_subparticle_fields(self):
        """Expand the size of the subparticle fields."""
        num_curr = self.sub_props['masses'].shape[0]
        print(f"Expanding subparticle arrays, current size: {num_curr}")
        for key in self.sub_props:
            new_arr = np.zeros(self.sub_props[key].shape,
                dtype=self.sub_props[key].dtype)
            self.sub_props[key] = np.concatenate((self.sub_props[key], new_arr))
        self.num_allocated_sub = self.sub_props['masses'].shape[0]
        print(f"Done, now place for {self.num_allocated_sub} sub-parts.")

    def shrink_subparticle_arrays(self):
        """Shrink subparticle arrays to actual number of generated particles."""
        print(f"Shrinking subparticle arrays from "
              f"{self.sub_props['masses'].shape[0]} to {self.num_sub} entries.")
        for key in self.sub_props:
            self.sub_props[key] = self.sub_props[key][:self.num_sub, ...]

if __name__ == "__main__":
    main()