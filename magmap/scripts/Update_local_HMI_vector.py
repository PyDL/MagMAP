"""
Our HMI-vector maps essentially use the vector data for active regions, and
scaled magnetograms for the remaining map.  All this to say that the vector
product depends on magnetogram availability.
# So use the existing HMI-M
# raw disk image file-system, and check for matching (in time) vector
# segments to download.
Update: make the vector data storage independent of the LOS storage.
Summary of functionality:
- Read existing filesystem HMI-B
- Determine timestamps to search at
- If all segments available, download
- Re-read filesystem and generate index csv
"""

import os
import numpy as np
import pandas as pd
import datetime

import pytz

import magmap.data.download.drms_helpers as drms_helpers
import magmap.utilities.file_io.io_helpers as io_helpers

# ---- Inputs -----------------------------
# select the data series (see notes at the top about the filter)
seriesM = 'hmi.m_720s'
seriesB = 'hmi.b_720s'

raw_dirname_M = '/Volumes/extdata3/oft/raw_data/hmi_m720s'
filters_M = ['QUALITY=0', ]

raw_dirname_B = '/Volumes/extdata3/oft/raw_data/hmi_b720s'
filters_B = ['QUALITY=0', ]
segments_B = ["field", "inclination", "disambig", "azimuth", "conf_disambig"]

# define image search interval cadence and width
interval_cadence = datetime.timedelta(hours=1)
del_interval = datetime.timedelta(minutes=20)
target_minute = 48
# set these to None for automatic updates
fixed_period_start = None
fixed_period_end = datetime.datetime(2024, 2, 1, 1, target_minute, 0, tzinfo=datetime.timezone.utc)

# index file name
index_file = "all-files.csv"

# ----- End Inputs -------------------------

# read data directory (would be faster—but less robust—to read existing index files)
robust = False
if robust:
    available_raw_B = io_helpers.read_db_dir(raw_dirname_B)
else:
    available_raw_B = pd.read_csv(raw_dirname_B + os.sep + "index_files" + os.sep + index_file)
    available_raw_B['date'] = pd.to_datetime(available_raw_B['date'], format='%Y-%m-%dT%H:%M:%S',
                                             utc=True)

# define date range to search
# initial time
if len(available_raw_B) > 0:
    # determine most recent data
    period_start = available_raw_B.date.max().to_pydatetime()
else:
    period_start = datetime.datetime(2010, 5, 1, 0, target_minute, 0, tzinfo=datetime.timezone.utc)

# optionally override the start time to force a re-examination
# period_start = datetime.datetime(2010, 5, 1, 0, target_minute, 0, tzinfo=datetime.timezone.utc)

# ending time
period_end = datetime.datetime.now(datetime.timezone.utc)

# override with fixed period datetimes
if fixed_period_start is not None:
    period_start = fixed_period_start
if fixed_period_end is not None:
    period_end = fixed_period_end

period_range = [period_start, period_end]

# define target times over download period using interval_cadence (image times in astropy Time() format)
if target_minute >= 30:
    tm_comp = target_minute - 30
else:
    tm_comp = target_minute + 30
rounded_start = period_start.replace(second=0, microsecond=0, minute=target_minute, hour=period_start.hour) + \
                datetime.timedelta(hours=period_start.minute//tm_comp - 1)
target_times = pd.date_range(start=rounded_start, end=period_end, freq=interval_cadence).to_pydatetime()
query_range = [target_times[0]-del_interval, target_times[-1]+del_interval]

# initialize the helper class for HMI
hmi = drms_helpers.HMI_M720s(verbose=True, series=seriesM, filters=filters_M)
hmi_vec = drms_helpers.HMI_B720s(verbose=True, filters=filters_B)

# query available magnetograms
available_hmi = hmi.query_time_interval(time_range=query_range)
available_vec = hmi_vec.query_time_interval(time_range=query_range, segments=segments_B)

# check vector query for missing segments
url_block = available_vec.loc[:, segments_B]
# not sure what a bad url will look like...
# but we should remove those rows
# available_vec = available_vec.loc[good_rows, :]

# generate DataFrame that defines synchronic target times as well as min/max limits
match_times = pd.DataFrame({'target_time': target_times, 'hmi_time': target_times})
# match closest available to each target time
available_los_datetimes = np.array([
    datetime.datetime.strptime(x, "%Y.%m.%d_%H:%M:%S").replace(tzinfo=pytz.UTC)
    for x in available_hmi.time])
available_vec_datetimes = np.array([
    datetime.datetime.strptime(x, "%Y.%m.%d_%H:%M:%S").replace(tzinfo=pytz.UTC)
    for x in available_vec.time])

for index, row in match_times.iterrows():
    # determine times where images are available for both HMI-M and HMI-B
    target_time = row.target_time.to_pydatetime()
    los_elig_index = np.where((available_los_datetimes >= (target_time-del_interval)) &
                              (available_los_datetimes <= (target_time+del_interval)))[0]
    los_elig = available_los_datetimes[los_elig_index]
    vec_elig_index = np.where((available_vec_datetimes >= (target_time-del_interval)) &
                              (available_vec_datetimes <= (target_time+del_interval)))[0]
    vec_elig = available_vec_datetimes[vec_elig_index]
    elig_int, los_ind, vec_ind = np.intersect1d(los_elig, vec_elig, return_indices=True)

    if len(elig_int) > 0:
        # find time closest to target time
        time_diff = elig_int - target_time
        time_diff = np.abs(time_diff)
        best_match = time_diff.argmin()

        # recover corresponding rows from los and vec
        los_row = available_hmi.loc[los_elig_index[los_ind[best_match]]]
        vec_row = available_vec.loc[vec_elig_index[vec_ind[best_match]]]

        # download resulting magnetogram
        print(f'Acquiring HMI-M data for time: {los_row.time}, quality: {los_row.quality}')
        sub_dir, fname, exit_flag = hmi.download_image_fixed_format(
            data_series=los_row, base_dir=raw_dirname_B, suffix="magnetogram",
            update=True, overwrite=False, verbose=True
        )

        # download resulting vector segments
        print(f'Acquiring HMI-B data for time: {los_row.time}, quality: {los_row.quality}')
        sub_dir_b, fname_b, exit_flag_b = hmi_vec.download_image_fixed_format(
            data_series=vec_row, base_dir=raw_dirname_B, segments=segments_B,
            update=True, overwrite=False, verbose=True
        )
        if any(exit_flag_b != 0):
            # If not all segments downloaded successfully, clean-up the remainder
            print("Cleaning-up any partial segment files for this timestamp.")
            hmi_vec.cleanup_download_image_fixed(data_series=vec_row, base_dir=raw_dirname_B,
                                                 segments=segments_B+["magnetogram", ])

    else:
        print(f'  NO SUITABLE DATA FOR INTERVAL AT: {row.hmi_time}')

# read the updated filesystem
available_raw = io_helpers.read_db_dir(raw_dirname_B)

# write to csv
index_full_path = os.path.join(raw_dirname_B, "index_files", index_file)
available_raw.to_csv(index_full_path, index=False,
                     date_format="%Y-%m-%dT%H:%M:%S", float_format='%.5f')
