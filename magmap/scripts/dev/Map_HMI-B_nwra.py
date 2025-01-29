import datetime
# Script to convert HMI-vector segments to a Br map

import os
import re
import shutil
import time

import numpy as np
from astropy.time import Time

import magmap.utilities.datatypes.datatypes as psi_dt
import magmap.maps.hipft_prep as hipft_prep
import magmap.maps.util.map_manip as map_manip
import magmap.utilities.file_io.io_helpers as io_helpers

# various paths
raw_dat_dir = "/Volumes/extdata3/oft/raw_data/raw_download_nwra/pub/PSI"
map_dat_dir = "/Volumes/extdata3/oft/processed_maps/hmi-b_nwra"

# input data filename format
raw_dat_form = "hmi.b_720s.DDD_TTT_TAI.field.fits"
input_file_format = "nwra"
# output map filename format
map_out_form = "hmi-b_map_720s_ymdThms.h5"
# index file name
index_file = "all-files.csv"
index_file_ron = "magmap_maps_all.csv"

# Input files are collected into directories by month
proc_year = (2011, )
proc_month = (4, 5, 6)

# number of parallel processors and threads-per-processor
nprocs = 1
tpp = 5
print_timings = True

# High-res map grid specifications (interpolation grid)
map_nxcoord = 10240
map_nycoord = 5120
R0 = 1.

# reduced map grid specifications
reduced_nxcoord = 1024
reduced_nycoord = 512

# ----- End Inputs -------------------------

# initiate a pool of processors
# p_pool = Pool(nprocs)
p_pool = None

# setup map grid
y_range = [-np.pi/2, np.pi/2]
x_range = [0, 2 * np.pi]
x_axis = np.linspace(x_range[0], x_range[1], map_nxcoord)
y_axis = np.linspace(y_range[0], y_range[1], map_nycoord)
# adjust polar-coordinate interpolation-centers for half-pixel mesh
dy = y_axis[1] - y_axis[0]
y_interp = y_axis.copy()
y_interp[0] = y_interp[0] + dy/4
y_interp[-1] = y_interp[-1] - dy/4
# interp expects sin(lat)
sin_lat = np.sin(y_interp)

# setup reduced-map grid (for saving to file)
reduced_x = np.linspace(x_range[0], x_range[1], reduced_nxcoord)
reduced_y = np.linspace(y_range[0], y_range[1], reduced_nycoord)
# interp expects sin(lat)
reduced_sin_lat = np.sin(reduced_y)

# initialize timing variables
IOtime = 0
image_proc_time = 0
interp_time = 0
down_samp_time = 0
map_proc_time = 0
n_its = 0

# loop through years
for year in proc_year:
    map_year_dir = os.path.join(map_dat_dir, str(year))
    # create directory if needed
    if not os.path.exists(map_year_dir):
        os.makedirs(map_year_dir, mode=0o755)
    for month in proc_month:
        dat_month_dir = os.path.join(raw_dat_dir, str(year) + str(month).zfill(2))
        map_month_dir = os.path.join(map_year_dir, str(month).zfill(2))
        # create directory if needed
        if not os.path.exists(map_month_dir):
            os.makedirs(map_month_dir, mode=0o755)

        # now read the filenames, and extract all timestamps
        all_paths = os.walk(dat_month_dir)
        root, dirs, files = all_paths.__next__()
        # Regular expression pattern for the timestamp
        pattern = r"\d{4}\.\d{2}\.\d{2}_\d{2}:\d{2}:\d{2}"
        # Extract timestamps from each string
        timestamp_str = [re.findall(pattern, string) for string in files]
        # Flatten list and remove empty sublists if necessary
        timestamp_str = [ts for sublist in timestamp_str for ts in sublist]
        timestamp_str = np.unique(timestamp_str)

        # loop through timestamps
        for ts in timestamp_str:
            # Convert to Astropy Time object
            timestamp = Time(ts.replace(".", "-").replace("_", "T"), format='isot', scale='tai', out_subfmt='date_hms')
            # Open data into a VectorMagneto object
            start_time = time.time()
            field_path = os.path.join(dat_month_dir, raw_dat_form.replace("DDD_TTT", ts))
            print("Loading segments of HMI-B for timestamp: " + str(timestamp))
            load_flag, vec_disk = psi_dt.read_hmi720s_vector(fits_file=field_path, file_format=input_file_format)
            IOtime += time.time() - start_time
            if load_flag == 1:
                print("Cannot load vector data. Proceeding to next timestamp.")
                continue

            # calculate Stonyhurst heliographic coordinates for each pixel
            start_time = time.time()
            print("Converting segments to Br and merging with LOS data.")
            vec_disk.get_coordinates(R0=R0, helio="SH")
            # Calculate Br
            vec_disk.calc_Br(R0=R0)
            image_proc_time += time.time() - start_time

            # Interpolate to CR map
            start_time = time.time()
            hmi_map = vec_disk.interp_to_map(R0=R0, map_x=x_axis, map_y=sin_lat, interp_field="Br",
                                             nprocs=nprocs, tpp=tpp, p_pool=p_pool, no_data_val=-65500.,
                                             y_cor=False, helio_proj=True)
            interp_time += time.time() - start_time

            # Downsample to HipFT resolution
            start_time = time.time()
            reduced_map = map_manip.downsamp_reg_grid(full_map=hmi_map, new_y=reduced_sin_lat, new_x=reduced_x,
                                                      image_method=0,
                                                      periodic_x=True, y_units='sinlat', uniform_poles=True,
                                                      uniform_no_data=True)
            down_samp_time += time.time() - start_time

            # assign map y-axis back to phi
            reduced_map.y = reduced_y
            # set assimilation weights
            reduced_map = hipft_prep.set_assim_wghts(reduced_map, assim_method="mu4_upton")

            # Save map to HDF5
            start_time = time.time()
            mday = timestamp.to_datetime().day
            map_day_dir = os.path.join(map_month_dir, str(mday).zfill(2))
            # create directory if needed
            if not os.path.exists(map_day_dir):
                os.makedirs(map_day_dir, mode=0o755)
            map_filename = map_out_form.replace("ymdThms", timestamp.to_datetime().strftime("%Y%m%dT%H%M%S"))
            rel_path = os.path.join(str(year), str(month).zfill(2), str(mday).zfill(2), map_filename)
            reduced_map.write_to_file(map_dat_dir, map_type='magneto-vector', filename=rel_path)
            IOtime += time.time() - start_time

            print("\n")
            n_its += 1

# read all filenames and create a new index file
hipft_index = io_helpers.gen_hipft_index(map_dat_dir)
# write to csv
index_full_path = os.path.join(map_dat_dir, "index_files", index_file)
hipft_index.to_csv(index_full_path, index=False,
                   date_format="%Y-%m-%dT%H:%M:%S", float_format='%.5f')
# copy to Ron's filename specification
index2_full_path = os.path.join(map_dat_dir, "index_files", index_file_ron)
shutil.copy2(index_full_path, index2_full_path)


print_timings = True
if print_timings:
    IOtime = IOtime/n_its
    image_proc_time = image_proc_time/n_its
    interp_time = interp_time/n_its
    down_samp_time = down_samp_time/n_its
    map_proc_time = map_proc_time/n_its

    total_time = IOtime + image_proc_time + interp_time + down_samp_time + map_proc_time

    print("Timings - averages per map:")
    print("Total time: ", "{:.2f}".format(total_time))
    print("IO time: ", "{:.2f}".format(IOtime), "s or ",
          "{:.1f}".format(IOtime/total_time*100), "%", sep="")
    print("Image proc time: ", "{:.2f}".format(image_proc_time), "s or ",
          "{:.1f}".format(image_proc_time/total_time*100), "%", sep="")
    print("Interpolation time: ", "{:.2f}".format(interp_time), "s or ",
          "{:.1f}".format(interp_time/total_time*100), "%", sep="")
    print("Down sampling time: ", "{:.2f}".format(down_samp_time), "s or ",
          "{:.1f}".format(down_samp_time/total_time*100), "%", sep="")
    print("Map proc time: ", "{:.2f}".format(map_proc_time), "s or ",
          "{:.1f}".format(map_proc_time/total_time*100), "%", sep="")
