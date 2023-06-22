#!/bin/bash

# Batch file to process Hydrangea galaxies through SKIRT
# Optional argument is the file name containing the subhalo list to process
# (otherwise it is generated internally with `make_galaxy_list.py`)

isim=21
isnap=27

coda="Lmax14_N3e7"

# -------------------------------------------------------------------

# Adapt these two paths to your system (location of python and skirt)
python="/net/watering/data1/anaconda3/bin/python3"
skirt="/net/quasar/data2/bahe/SKIRT/release/SKIRT/main/skirt"
rootdir="/data2/bahe/SKIRT/Hydrangea"   # output root directory

# ============================================================================

$python --version

printf -v isnap_str "%03d" $isnap
basedir="${rootdir}/CE-${isim}/Snapshot_${isnap_str}/extended/"
echo "Base directory is ${basedir}"
wdir=`pwd`

# Set up output directory
img_dir="${basedir}/Images"
if [ ! -d "${img_dir}" ]; then
    mkdir -p "${img_dir}/Images"
fi

if [ $# -ge 1 ]; then
    subhalo_list=$1
else
    # We need to build a subhalo list ourselves...
    subhalo_list="${basedir}/subhaloes_CE-${isim}_Snapshot-${isnap}.txt"
    $python make_galaxy_list.py \
	    -c $isim -x $isnap -m 1e10 1e20 \
	    -f $subhalo_list -d 2.0
fi

counter=0
skip_num=0
num_tot=100000000

while read line; do
    # Skip comment lines
    if [ ${line:0:1} == '#' ]; then
	
	# TODO: insert test for sim number and snapshot here...
	if [[ $line == *"[TEST STRING]"* ]]; then
  	    echo "It's there!"
        fi
	continue
    fi

    # Skip first X lines if desired
    if [ ${counter} -lt ${skip_num} ]; then
	counter=$((counter+1))
	continue
    fi

    # End prematurely if desired
    if [ $counter -ge ${num_tot} ]; then
	break
    fi

    isub=$line

    # Ok, we have the subhalo to process now (`isub`). Extract data:
    cd $wdir
    subhalo_dir="${basedir}/S-${isub}_${coda}"
    echo "${subhalo_dir}"
    $python ./extract_data_from_hydrangea.py \
	    -c $isim -x $isnap -s $isub -d "${subhalo_dir}"
    
    # Copy and adapt SKI-file template
    prefix="CE-${isim}_Snapshot-${isnap}_S-${isub}"
    ski_file="${prefix}.ski"
    fits_file="${prefix}_img-z_total.fits"

    cp ski_template_${coda}_split-output.ski ${subhalo_dir}/${ski_file}
    cd ${subhalo_dir}
    sed -i "s/XXX-STAR_FILE-XXX/sim_stars.${isim}.${isnap}.${line}.txt/g" \
	$ski_file
    sed -i "s/XXX-DUST_FILE-XXX/sim_dust.${isim}.${isnap}.${line}.txt/g" \
	$ski_file
    sed -i "s/XXX-HII_FILE-XXX/sim_hii.${isim}.${isnap}.${line}.txt/g" \
	$ski_file

    # Run SKIRT
    $skirt $ski_file

    # Make image
    cd $wdir
    $python make_image_from_skirt.py -i ${subhalo_dir}/${fits_file} \
	    -s 30

    # Set up a link to the image
    ln -s "${subhalo_dir}/${prefix}_img-z_total.sdss.png" \
       "${img_dir}/${prefix}_img-z_total.sdss.png"
 
    counter=$((counter+1))
    
done <${subhalo_list}
