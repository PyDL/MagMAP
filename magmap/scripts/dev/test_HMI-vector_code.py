
# Test script for reading and processing HMI vector data (hmi_b_720s) to a radial flux map


import magmap.utilities.datatypes.datatypes as psi_dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# file_format = 'PSI'
# field_segment_path = '/Users/turtle/Dropbox/MyOFT/hmi_vector/hmi_b720s_examples/2012/06/14/HMI_B_720__20120614_015826_field.fits'

file_format = "nwra"
# field_segment_path = '/Volumes/extdata3/oft/raw_data/raw_download_nwra/pub/PSI/201101/hmi.b_720s.2011.01.01_00:48:00_TAI.field.fits'
field_segment_path = '/Volumes/extdata3/oft/raw_data/raw_download_nwra/pub/PSI/201101/hmi.b_720s.2011.01.18_17:48:00_TAI.field.fits'

R0 = 1.

# load data segments to VectorMagneto object
load_flag, vec_disk = psi_dt.read_hmi720s_vector(fits_file=field_segment_path, file_format=file_format)

# calculate Stonyhurst heliographic coordinates for each pixel
vec_disk.get_coordinates(R0=R0, helio="SH")

def vec_plot(vec_disk, plot_vec, use_index=None):
    im_dims = vec_disk.data.shape
    vec_n = im_dims[0]*im_dims[1]

    plot_mat = np.full(vec_n, 0.)

    if len(plot_vec) < vec_n:
        if use_index is None:
            raise Exception("Short vectors require 'use_index' input.")

        plot_mat[use_index] = plot_vec
    else:
        plot_mat = plot_vec

    plot_mat = plot_mat.reshape(im_dims, order="C")

    plt.imshow(plot_mat, origin='lower', vmin=plot_vec.min(), vmax=plot_vec.max())
    plt.colorbar()
    plt.show()

# test plots to verify coordinates look correct
use_index = vec_disk.lat > vec_disk.no_data_val
plt.imshow(vec_disk.mu, origin="lower", aspect="equal", vmin=0., vmax=1., interpolation="none")
plt.colorbar()
plt.show()

# Solar north is very close to down in HMI
plt.imshow(vec_disk.lat, origin="lower", aspect="equal", vmin=-np.pi/2, vmax=np.pi/2, interpolation="none")
plt.colorbar()
plt.show()

# Assuming Stonyhurst, phi=0 should be centered, and phi should increase to the left
plt.imshow(vec_disk.lon, origin="lower", aspect="equal", vmin=0., vmax=2*np.pi, interpolation="none")
plt.colorbar()
plt.show()


# --- Calculate Br from the segments ---------------
vec_disk.calc_Br(R0=R0)

plt.imshow(vec_disk.Br, origin="lower", aspect="equal", vmin=-25, vmax=25, interpolation="none",
           cmap="seismic")
plt.colorbar()
plt.show()



