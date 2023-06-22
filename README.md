# skirt-processing
Tools to work with SKIRT for image generation

It contains the following scripts:

- `extract_data_from_hydrangea.py`: Extract data for one galaxy from Hydrangea output into SKIRT-compatible format. Currently uses only Subfind centering and hard-coded 50 kpc extraction aperture.

- `skirt_batch.sh`: Bash script to auto-process a full list of subhaloes through SKIRT (optionally finding these internally for a given Sim/Snap)

- `make_galaxy_list.py`: Find galaxies in a simulation depending on a set of criteria

- `make_image_from_skirt.py`: Generate an image from the SKIRT output.
