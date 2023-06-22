"""Generate a gri image from SKIRT output."""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import argparse
from pdb import set_trace

def main():

    parser = argparse.ArgumentParser(
        description="Make a gri image from a SKIRT simulation output.")
    parser.add_argument(
        '-i', '--input', help="SKIRT output file", required=True)
    parser.add_argument(
        '-s', '--imsize', type=float, help="Image half-sidelength in kpc.",
        required=True)
    parser.add_argument(
        '-r', '--range', type=float, nargs='*', default=[26, 18],
        help="Scaling range for image, in mag per square arcsec."
    )
    args = parser.parse_args()

    # Open FITS file containing the SKIRT image
    hdul = fits.open(args.input)
    cube = hdul[0].data
    hdr = hdul[0].header

    # Conversion factor from SKIRT units (MJy/Sr) to ergs s^-1 cm^-2 Hz^-1 arcsec^-2
    if hdr['BUNIT'] != 'MJy/sr':
        raise ValueError(f"SKIRT file in unit of {hdr['BUNIT']}!")

    # MJy -> Jy * Jy -> W Hz^-1 m^-2 * m^-2 -> cm^-2 * W -> erg / Sr -> arcsec^2
    f_conv = 1e6 * 1e-26 * 1e-4 * 1e7 / (4.255e10) 
    print(f"Conversion factor: {f_conv:.4e}.")

    # Fluxes in the g, r, i band, in ergs s^-1 cm^-2 Hz^-1 arcsec^-2
    flux_g = cube[1, :, :] * f_conv
    flux_r = cube[2, :, :] * f_conv
    flux_i = cube[3, :, :] * f_conv

    # Convert the scaling range to the same linear units
    range_f = flux(np.array(args.range))

    make_image(flux_i, flux_r, flux_g, range_f, args,
               type='standard', suffix='std')
    make_image(flux_i, flux_r, flux_g, range_f, args,
               type='lupton', suffix='sdss')


def make_image(r, g, b, limits, args, type='standard', suffix='std'):
    """Make an RGB image from three colour channels."""

    nxpix, nypix = r.shape

    if type.lower() == 'lupton':
        # SDSS-like method, based on Lupton et al. 2004 (PASP, 116, 133)
        t = (r + g + b) / 3
        lum = scale(t, limits)

        rgb = np.zeros((nxpix, nypix, 3))
        rgb[:, :, 0] = lum / t * r
        rgb[:, :, 1] = lum / t * g
        rgb[:, :, 2] = lum / t * b

        ind_low = np.nonzero(t == 0)
        rgb[ind_low[0], ind_low[1], :] = 0

        maxrgb = np.max(rgb, axis=2)
        ind_high = np.nonzero(maxrgb > 1)
        for ichannel in range(3):
            rgb[ind_high[0], ind_high[1], ichannel] /= (
                maxrgb[ind_high[0], ind_high[1]])

    else:
        # Standard log scaling method
        rgb = np.zeros((nxpix, nypix, 3))
        rgb[:, :, 0] = scale(r, limits)
        rgb[:, :, 1] = scale(g, limits)
        rgb[:, :, 2] = scale(b, limits)

    # Display and save the RGB image
    fig = plt.figure(figsize=(4, 4))

    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(rgb, extent=np.array([-1, 1, -1, 1]) * args.imsize,
              origin='lower')

    phivec = np.arange(0, 2.001*np.pi, 0.01)
    plt.plot(30 * np.cos(phivec), 30 * np.sin(phivec), color='grey',
        linestyle=':', linewidth=0.5)

    outfile = args.input.replace('.fits', f'.{suffix}.png')
    plt.savefig(outfile, dpi=300)
    print(f"Saved image at {outfile}.")


def ab(flux):
    return -2.5 * np.log10(flux + 1e-40) - 48.6

def flux(ab):
    return 10**(-0.4 * (ab + 48.6))

def scale(x, limits):
    return np.clip((np.log10(x) - np.log10(limits[0])) /
                   (np.log10(limits[1]) - np.log10(limits[0])),
                   0, 1
    )

if __name__ == "__main__":
    main()