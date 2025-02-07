"""
Helper routines for working with the SDO JSOC DRMS system.

Routines specific to the aia.lev1_euv_12s are held in their own class
- various capabilities include
  - update the header info of "as-is" fits files downloaded from the JSOC
    - use a user supplied fits header to query the JSOC via drms
    - Updates the header with the info returned from the JSOC and re-compresses it
  - build time queries based of of a sunpy time_range
- This uses the astropy fits interface and the JSOC drms package
- The series specific class sets up the DRMS client on initialization
- Modifications for other series are probably required.
"""
import astropy.io.fits
import astropy.time
import astropy.units
import sunpy
import drms
import math
import os
import numpy as np
import pandas as pd
from http.client import HTTPException
from collections import OrderedDict
import requests
import time

from magmap.utilities.file_io import io_helpers

# ----------------------------------------------------------------------
# global definitions
# ----------------------------------------------------------------------

# UTC postfix
time_zone_postfix = 'Z'

# SDO/JSOC web location
jsoc_url = 'http://jsoc.stanford.edu'


class HMI_M720s:
    """
    Class that holds drms/JSOC specific routines for the hmi.m_720s series
    - On initialization it sets up a client that will be used to work with this data
    """

    def __init__(self, series='hmi.m_720s', filters=['QUALITY=0', ], verbose=False):

        self.series = series

        # initialize the drms client
        self.client = drms.Client()

        # obtain ALL of the keys for this series (i believe this connects to JSOC)
        self.allkeys = self.client.keys(self.series)

        # ---------------------------------------------
        # define the series specific default variables here
        # ---------------------------------------------
        # default keys for a normal query over a range of times
        # self.default_keys = ['**ALL**', ]
        self.default_keys = ['T_REC', 'T_OBS', 'EXPTIME', 'CAMERA', 'QUALITY']

        # default filters to use for a query
        self.default_filters = filters
        # self.default_filters = ['QUALITY=0', ]
        # self.default_filters = ['CAMERA=1', 'QUALITY=0']
        # self.default_filters = ['QUALITY=1024 or QUALITY=72704']

        # default segments for a query
        self.default_segments = ['magnetogram']

        # header items that change from "as-is" to "fits" formats served by JSOC.
        self.hdr_keys_to_delete = ['SOURCE', ]
        # self.hdr_keys_to_add = ['T-OBS', 'CAMERA', 'T-OBS-JD', 'SPC-CRFT',
        #                         'INST']
        self.hdr_keys_to_add = ['ALL', ]

        if verbose:
            print('### Initialized DRMS client for ' + self.series)

    def update_hmi_fits_header(self, infile, drms_query, verbose=False, force=False):
        """
        read an HMI fits file, update the header, save the file
        The silentfix is to avoid warnings/and or crashes when astropy encounters nans in the header values
        """

        # open file for update
        hdu_in = astropy.io.fits.open(infile, mode='update')
        # The silentfix is to avoid warnings/and or crashes when astropy encounters nans in the header values
        hdu_in.verify('silentfix')
        hdr = hdu_in[1].header

        # check to see if this file has already been converted
        if set(self.hdr_keys_to_add).issubset(hdr) and not set(self.hdr_keys_to_delete).issubset(hdr) and not force:
            if verbose:
                print("### " + infile + " looks like it has an updated header. SKIPPING.")
            return

        # get the corresponding header info from the JSOC as a drms frame
        drms_frame = self.get_drms_info_for_image(drms_query)

        # update the header info
        hdr_update = self.update_header_fields_from_drms(hdr, drms_frame,
                                                         verbose=verbose)

        # update header and close
        hdu_in[1].header = hdr_update
        hdu_in.close()


    def get_drms_info_for_image(self, init_query):
        """
        supply a single row from the result of query_time_interval(),
        obtain the full JSOC info as a pandas frame from drms
        The prime key syntax for hmi.m_720s is used here
        """
        query_string = '%s[%s_TAI][%s]'%(self.series, init_query.time.iloc[0], init_query.camera.iloc[0])
        drms_frame = self.client.query(query_string, key="**ALL**")
        # remove unwanted fields
        all_cols = set(drms_frame.keys())
        keep_cols = all_cols.difference(set(self.hdr_keys_to_delete))

        # get the frame matching these keywords, apparently keep_cols must be a list now
        drms_frame = drms_frame.loc[:, list(keep_cols)]

        return drms_frame

    def update_header_fields_from_drms(self, hdr, drms_frame, verbose=False):
        """
        update the fits header using info from a drms pandas frame
        This works for converting "as-is" hmi.m_720s headers to the "fits" style
        """

        # Replace all NaNs with a flag value
        if "BLANK" in drms_frame.keys():
            blank_val = drms_frame.BLANK[0]
        else:
            blank_val = -2147483648
        drms_frame.fillna(blank_val, inplace=True)

        # Delete the unwanted "as-is" protocol keys
        if self.hdr_keys_to_delete is not None:
            for key in self.hdr_keys_to_delete:
                if key in hdr:
                    hdr.remove(key)

        # setup the conversion of some drms pandas tags to "fits" tags
        hdr_pandas_keys_to_fits = {
            'DATE__OBS': 'DATE_OBS',
            'T_REC_epoch': 'TRECEPOC',
            'T_REC_step': 'TRECSTEP',
            'T_REC_unit': 'TRECUNIT'}

        # Add a history line about the conversion
        my_history = 'Updated the JSOC "as-is" ' + self.series + ' header information to "fits" using DRMS query info.'
        hdr.add_history(my_history)

        # loop over the drms keys, update the corresponding FITS header tags
        for pandas_key in drms_frame.keys():
            fits_key = pandas_key

            if pandas_key in hdr_pandas_keys_to_fits:
                fits_key = hdr_pandas_keys_to_fits[pandas_key]

            # normal behavior fits key already exists, update w/ same time.
            if fits_key in hdr:
                fits_val = hdr[fits_key]

                pandas_val = drms_frame[pandas_key][0]
                # remove line breaks from strings
                if type(pandas_val) is str:
                    pandas_val = pandas_val.replace("\n", "")
                # trap for nan ints to maintain the standard types JSOC uses in their fits files.
                if type(fits_val) is int:
                    if math.isnan(pandas_val):
                        pandas_val = blank_val
                elif pandas_key == "HISTORY":
                    hdr.add_history(pandas_val)
                elif pandas_key == "COMMENT":
                    hdr.add_comment(pandas_val)
                else:
                    pandas_val = drms_frame[pandas_key].astype(type(fits_val))[0]

                if fits_val != pandas_val:
                    if verbose:
                        print('Diff Val: ', pandas_key, fits_key, drms_frame[pandas_key][0], hdr[fits_key])
                    hdr[fits_key] = pandas_val

            # Check for adding a key that doesn't exist.
            elif fits_key in self.hdr_keys_to_add or (len(self.hdr_keys_to_add) == 1
                                                      and self.hdr_keys_to_add[0] == "ALL"):
                hdr[fits_key] = drms_frame[pandas_key][0]

            # Check if the key is missing from the file and wasn't supposed to be added.
            else:
                if verbose:
                    print('MISSING:   ', pandas_key, drms_frame[pandas_key][0])

        # Last, check for 'DATE-OBS' field and convert to MJD
        if "DATE_OBS" in hdr.keys():
            date_str = hdr["DATE-OBS"]
            date_str = date_str.replace("Z", "")
            dtime = astropy.time.Time(date_str, format='fits')
            mjd = dtime.mjd
            hdr['MJD-OBS'] = mjd

        return hdr

    def query_time_interval(self, time_range, search_cadence=None,
                            filters=None, segments=None, keys=None):
        """
        Quick function to query the JSOC for all matching magnetograms over
        a certain interval (intervals should be specified in TAI time)
        - returns the drms pandas dataframes of the keys and segments
        - if no magnetograms, len(keys) and len(segs) will be zero
        """

        # set up the default values (these should never be passed as None anyway)
        if filters is None:
            filters = self.default_filters
        if segments is None:
            segments = self.default_segments
        if keys is None:
            keys = self.default_keys

        # build the keys string
        key_str = ', '.join(keys)

        # build the Filter Query String
        if len(filters) > 0:
            filter_str = '[? ' + ' ?][? '.join(filters) + ' ?]'
        else:
            filter_str = ''

        # build the segments string
        segment_str = '{' + ', '.join(segments) + '}'

        # build the time interval string
        if search_cadence is not None:
            if type(search_cadence) is astropy.units.quantity.Quantity:
                cadence_str = '@' + get_jsoc_interval_format(search_cadence)
            else:
                cadence_str = '@' + search_cadence
        else:
            cadence_str = ""

        # keep back-compatibility, but convert time_range to a list of datetimes
        if type(time_range) is sunpy.time.timerange.TimeRange:
            time_start = time_range.start.to_datetime()
            time_end = time_range.end.to_datetime()
        else:
            time_start = time_range[0]
            time_end = time_range[1]
        time_range_str = time_start.strftime("%Y.%m.%d_%H:%M:%S") + \
            "_TAI-" + time_end.strftime("%Y.%m.%d_%H:%M:%S") + "_TAI"
        time_query_str = "[" + time_range_str + cadence_str + "]"

        # build the query
        query_string = self.series + time_query_str + filter_str + segment_str

        # query JSOC for image times
        try:
            key_frame, seg_frame = self.client.query(query_string, key=key_str, seg=segments)
        except HTTPException:
            print("There was a problem contacting the JSOC server to query image times. Trying again...\n")
            try:
                key_frame, seg_frame = self.client.query(query_string, key=key_str, seg=segments)
            except HTTPException:
                print("Still cannot contact JSOC server. Returning 'query error'.")
                return "query error"

        if len(seg_frame) == 0:
            # No results were found for this time range. generate empty Dataframe to return
            data_frame = pd.DataFrame(columns=('spacecraft', 'instrument', 'camera', 'filter', 'quality', 'time', 'jd', 'url'))
            return data_frame

        # parse the results a bit
        time_strings, jds = parse_query_times(key_frame, time_type='tai')
        urls = jsoc_url + seg_frame['magnetogram']

        # make the custom dataframe with the info i want
        # data_frame = io_helpers.custom_dataframe(time_strings, jds, urls, 'SDO', 'HMI')
        data_frame = pd.DataFrame(
            OrderedDict({'spacecraft': 'SDO', 'instrument': 'HMI', 'camera': key_frame.CAMERA,
                         'filters': filter_str, 'quality': key_frame.QUALITY, 'time': time_strings, 'jd': jds, 'url': urls})
        )

        # return key_frame, seg_frame
        return data_frame

    def download_image_fixed_format(self, data_series, base_dir, suffix="",
                                    update=True, overwrite=False, verbose=False):
        """
        supply a row from a custom pandas data_frame and use this info to download the HMI image.
        (a pandas series is a row of a dataframe, so these are basically
        the single image results from the query)
        - you can obtain these by doing data_framge = key=keys.iloc[row_index]
        The sub_path and filename are determined from the image information
        exit_flag: 0-Successful download; 1-download error; 2-file already exists
        """
        if len(data_series.shape) != 1:
            raise RuntimeError('data_series has more than one row!')

        # build the url
        url = data_series['url']

        # build the filename and subdir from the series, timestamp, and wavelength information
        # image_datetime = astropy.time.Time(data_series['time'], scale='tai').datetime
        image_datetime = astropy.time.Time.strptime(
            data_series['time'], format_string="%Y.%m.%d_%H:%M:%S", scale='tai'
        ).datetime
        prefix = '_'.join(self.series.split('.'))
        # postfix = str(data_series['filter'])
        postfix = suffix
        ext = 'fits'
        dir, fname = io_helpers.construct_path_and_fname(base_dir, image_datetime, prefix, postfix, ext)
        fpath = dir + os.sep + fname

        # download the file
        exit_flag = io_helpers.download_url(url, fpath, verbose=verbose, overwrite=overwrite)

        if exit_flag == 1:
            # There was an error with download. Try again
            print(" Re-trying download....")
            exit_flag = io_helpers.download_url(url, fpath, verbose=verbose, overwrite=overwrite)
            if exit_flag == 1:
                # Download failed. Return None
                return None, None, exit_flag

        # update the header info if desired
        if update:
            if verbose:
                print('  Updating header info if necessary.')
            drms_query = data_series.to_frame().T
            self.update_hmi_fits_header(fpath, drms_query, verbose=False)

        print("")
        # now separate the the sub directory from the base path (i want relative path for the DB)
        sub_dir = os.path.sep.join(dir.split(base_dir)[1].split(os.path.sep)[1:])

        return sub_dir, fname, exit_flag


class HMI_B720s:
    """
    Class that holds drms/JSOC specific routines for the hmi.B vector data
    - On initialization it sets up a client that will be used to work with this data
    """

    def __init__(self, verbose=False, filters=None):

        self.series = 'hmi.b_720s'

        # initialize the drms client
        self.client = drms.Client()

        # obtain ALL of the keys for this series (i believe this connects to JSOC)
        self.allkeys = self.client.keys(self.series)

        # ---------------------------------------------
        # define the series specific default variables here
        # ---------------------------------------------
        # default keys for a normal query over a range of times
        # self.default_keys = ['**ALL**', ]
        self.default_keys = ['T_REC', 'T_OBS', 'EXPTIME', 'CAMERA', 'QUALITY']

        # default filters to use for a query
        self.default_filters = ['QUALITY=0', ]
        # self.default_filters = ['CAMERA=1', 'QUALITY=0']

        # default segments for a query
        self.default_segments = ["field", "inclination", "disambig", "azimuth", "conf_disambig"]

        # header items that change from "as-is" to "fits" formats served by JSOC.
        self.hdr_keys_to_delete = ['SOURCE', ]
        # self.hdr_keys_to_add = ['T-OBS', 'CAMERA', 'T-OBS-JD', 'SPC-CRFT',
        #                         'INST']
        self.hdr_keys_to_add = ['ALL', ]

        if verbose:
            print('### Initialized DRMS client for ' + self.series)

        if filters is not None:
            self.default_filters = filters

    def update_hmi_fits_header(self, infile, drms_query, verbose=False, force=False):
        """
        read an HMI fits file, update the header, save the file
        The silentfix is to avoid warnings/and or crashes when astropy encounters nans in the header values
        """

        # open file for update
        hdu_in = astropy.io.fits.open(infile, mode='update')
        # The silentfix is to avoid warnings/and or crashes when astropy encounters nans in the header values
        hdu_in.verify('silentfix')
        hdr = hdu_in[1].header

        # check to see if this file has already been converted
        if set(self.hdr_keys_to_add).issubset(hdr) and not set(self.hdr_keys_to_delete).issubset(hdr) and not force:
            if verbose:
                print("### " + infile + " looks like it has an updated header. SKIPPING.")
            return

        # get the corresponding header info from the JSOC as a drms frame
        drms_frame = self.get_drms_info_for_image(drms_query)

        if drms_frame is not None:
            # update the header info
            hdr_update = self.update_header_fields_from_drms(hdr, drms_frame,
                                                             verbose=verbose)
            # update header and close
            hdu_in[1].header = hdr_update
            update_flag = 0
        else:
            update_flag = 1

        hdu_in.close()

        return update_flag


    def get_drms_info_for_image(self, init_query):
        """
        supply a single row from the result of query_time_interval(),
        obtain the full JSOC info as a pandas frame from drms
        This is modeled off the HMI_M720s function, but with the following changes:
          - The camera portion of the query is removed (because it is not part of the
            HMI_B720s query format.
        """
        query_string = '%s[%s_TAI]'%(self.series, init_query.time.iloc[0])
        # drms_frame = self.client.query(query_string, key="**ALL**")
        drms_frame = drms_query_with_retry(self, query_string, keys="**ALL**",
                                           max_retries=5, delay=2)

        if drms_frame is not None:
            # remove unwanted fields
            all_cols = set(drms_frame.keys())
            keep_cols = all_cols.difference(set(self.hdr_keys_to_delete))
            drms_frame = drms_frame.loc[:, list(keep_cols)]

        return drms_frame

    def update_header_fields_from_drms(self, hdr, drms_frame, verbose=False):
        """
        update the fits header using info from a drms pandas frame
        This works for converting "as-is" hmi.m_720s headers to the "fits" style
        """

        # Replace all NaNs with a flag value
        if "BLANK" in drms_frame.keys():
            blank_val = drms_frame.BLANK[0]
        else:
            blank_val = -2147483648
        drms_frame.fillna(blank_val, inplace=True)

        # Delete the unwanted "as-is" protocol keys
        if self.hdr_keys_to_delete is not None:
            for key in self.hdr_keys_to_delete:
                if key in hdr:
                    hdr.remove(key)

        # setup the conversion of some drms pandas tags to "fits" tags
        hdr_pandas_keys_to_fits = {
            'DATE__OBS': 'DATE_OBS',
            'T_REC_epoch': 'TRECEPOC',
            'T_REC_step': 'TRECSTEP',
            'T_REC_unit': 'TRECUNIT'}

        # Add a history line about the conversion
        my_history = 'Updated the JSOC "as-is" ' + self.series + ' header information to "fits" using DRMS query info.'
        hdr.add_history(my_history)

        # loop over the drms keys, update the corresponding FITS header tags
        for pandas_key in drms_frame.keys():
            fits_key = pandas_key

            if pandas_key in hdr_pandas_keys_to_fits:
                fits_key = hdr_pandas_keys_to_fits[pandas_key]

            # normal behavior fits key already exists, update w/ same time.
            if fits_key in hdr:
                fits_val = hdr[fits_key]

                # trap for nan ints to maintain the standard types JSOC uses in their fits files.
                pandas_val = drms_frame[pandas_key][0]
                if type(fits_val) is int:
                    if math.isnan(pandas_val):
                        pandas_val = blank_val
                elif pandas_key == "HISTORY":
                    hdr.add_history(pandas_val)
                elif pandas_key == "COMMENT":
                    hdr.add_comment(pandas_val)
                else:
                    pandas_val = drms_frame[pandas_key].astype(type(fits_val))[0]

                if fits_val != pandas_val:
                    if verbose:
                        print('Diff Val: ', pandas_key, fits_key, drms_frame[pandas_key][0], hdr[fits_key])
                    hdr[fits_key] = pandas_val

            # Check for adding a key that doesn't exist.
            elif fits_key in self.hdr_keys_to_add or (len(self.hdr_keys_to_add) == 1
                                                      and self.hdr_keys_to_add[0] == "ALL"):
                # Keywords are required to be 8 characters or less by FITS standards
                # Auto-abbreviating keywords is not something we want to do.  Assume
                # longer keywords from the database are not standard FITS values and
                # exclude.
                if len(fits_key) <= 8:
                    hdr[fits_key] = drms_frame[pandas_key][0]

            # Check if the key is missing from the file and wasn't supposed to be added.
            else:
                if verbose:
                    print('MISSING:   ', pandas_key, drms_frame[pandas_key][0])

        # Last, check for 'DATE-OBS' field and convert to MJD
        if "DATE_OBS" in hdr.keys():
            date_str = hdr["DATE-OBS"]
            date_str = date_str.replace("Z", "")
            dtime = astropy.time.Time(date_str, format='fits')
            mjd = dtime.mjd
            hdr['MJD-OBS'] = mjd

        return hdr

    def query_time_interval(self, time_range, search_cadence=None,
                            filters=None, segments=None, keys=None):
        """
        Quick function to query the JSOC for all matching magnetograms over
        a certain interval (intervals should be specified in TAI time)
        - returns the drms pandas dataframes of the keys and segments
        - if no magnetograms, len(keys) and len(segs) will be zero
        """

        # set up the default values (these should never be passed as None anyway)
        if filters is None:
            filters = self.default_filters
        if segments is None:
            segments = self.default_segments
        if keys is None:
            keys = self.default_keys

        # build the keys string
        key_str = ', '.join(keys)

        # build the Filter Query String
        if len(filters) > 0:
            filter_str = '[? ' + ' ?][? '.join(filters) + ' ?]'
        else:
            filter_str = ''

        # build the segments string
        segment_str = '{' + ', '.join(segments) + '}'

        # build the time interval string
        if search_cadence is not None:
            if type(search_cadence) is astropy.units.quantity.Quantity:
                cadence_str = '@' + get_jsoc_interval_format(search_cadence)
            else:
                cadence_str = '@' + search_cadence
        else:
            cadence_str = ""

        # keep back-compatibility, but convert time_range to a list of datetimes
        if type(time_range) is sunpy.time.timerange.TimeRange:
            time_start = time_range.start.to_datetime()
            time_end = time_range.end.to_datetime()
        else:
            time_start = time_range[0]
            time_end = time_range[1]
        time_range_str = time_start.strftime("%Y.%m.%d_%H:%M:%S") + \
            "_TAI-" + time_end.strftime("%Y.%m.%d_%H:%M:%S") + "_TAI"
        time_query_str = "[" + time_range_str + cadence_str + "]"

        # build the query
        query_string = self.series + time_query_str + filter_str + segment_str

        # query JSOC for image times
        try:
            key_frame, seg_frame = self.client.query(query_string, key=key_str, seg=segments)
        except HTTPException:
            print("There was a problem contacting the JSOC server to query image times. Trying again...\n")
            try:
                key_frame, seg_frame = self.client.query(query_string, key=key_str, seg=segments)
            except HTTPException:
                print("Still cannot contact JSOC server. Returning 'query error'.")
                return "query error"

        if len(seg_frame) == 0:
            # No results were found for this time range. generate empty Dataframe to return
            data_frame = pd.DataFrame(columns=('spacecraft', 'instrument', 'camera', 'filter', 'time', 'jd')
                                              + tuple(segments))
            return data_frame

        # parse the results a bit
        time_strings, jds = parse_query_times(key_frame, time_type='tai')
        # urls = jsoc_url + seg_frame['magnetogram']

        # make the custom dataframe with the info i want
        # data_frame = io_helpers.custom_dataframe(time_strings, jds, urls, 'SDO', 'HMI')
        data_frame = pd.DataFrame(
            OrderedDict({'spacecraft': 'SDO', 'instrument': 'HMI', 'camera': key_frame.CAMERA,
                         'filters': filter_str, 'time': time_strings, 'jd': jds})
        )
        # add segment urls to data frame
        for seg in segments:
            data_frame[seg] = jsoc_url + seg_frame[seg]

        # return key_frame, seg_frame
        return data_frame

    def download_image_fixed_format(self, data_series, base_dir,
                                    segments=['field', 'inclination', 'disambig', 'azimuth', 'conf_disambig'],
                                    update=True, overwrite=False, verbose=False):
        """
        supply a row from a custom pandas data_frame and use this info to download the HMI image.
        (a pandas series is a row of a dataframe, so these are basically
        the single image results from the query)
        - you can obtain these by doing data_framge = key=keys.iloc[row_index]
        The sub_path and filename are determined from the image information
        exit_flag: 0-Successful download; 1-download error; 2-file already exists
        """
        if len(data_series.shape) != 1:
            raise RuntimeError('data_series has more than one row!')

        # initialize fname and exit_flag
        nsegs = len(segments)
        fname = np.full((nsegs, ), '', dtype=str)
        exit_flag = np.full((nsegs, ), 0, dtype=int)

        # loop through segments
        for ii in range(nsegs):
            seg = segments[ii]
            # extract the url
            url = data_series[seg]

            # build the filename and subdir from the series, timestamp, and wavelength information
            # image_datetime = astropy.time.Time(data_series['time'], scale='tai').datetime
            image_datetime = astropy.time.Time.strptime(
                data_series['time'], format_string="%Y.%m.%d_%H:%M:%S", scale='tai'
            ).datetime
            prefix = '_'.join(self.series.split('.'))
            # postfix = str(data_series['filter'])
            postfix = seg
            ext = 'fits'
            dir, seg_fname = io_helpers.construct_path_and_fname(base_dir, image_datetime, prefix, postfix, ext)
            fpath = dir + os.sep + seg_fname

            # download the file
            # seg_exit_flag = io_helpers.download_url(url, fpath, verbose=verbose, overwrite=overwrite)
            seg_exit_flag = io_helpers.download_url_with_requests(url, fpath, max_retries=5, delay=2,
                                                                  verbose=verbose, overwrite=overwrite)

            # if seg_exit_flag == 1:
            #     # There was an error with download. Try again
            #     print(" Re-trying download....")
            #     seg_exit_flag = io_helpers.download_url(url, fpath, verbose=verbose, overwrite=overwrite)
            if seg_exit_flag == 1:
                print(" Download of " + seg + " failed. Skipping remaining segments.")
                # Download failed. Exit for-loop and do clean-up
                exit_flag[ii] = seg_exit_flag
                break

            # update the header info if desired
            if update:
                if verbose:
                    print('  Updating header info if necessary.')
                drms_query = data_series.to_frame().T
                update_flag = self.update_hmi_fits_header(fpath, drms_query, verbose=False)
                if update_flag == 1:
                    print(" Header update of " + seg + " failed. Skipping remaining segments.")
                    exit_flag[ii] = update_flag
                    break

            # record fname and exit flag
            fname[ii] = seg_fname
            exit_flag[ii] = seg_exit_flag

        print("")
        # now separate the sub directory from the base path (i want relative path for the DB)
        sub_dir = os.path.sep.join(dir.split(base_dir)[1].split(os.path.sep)[1:])

        return sub_dir, fname, exit_flag


    def cleanup_download_image_fixed(self, data_series, base_dir,
                                    segments=['field', 'inclination', 'disambig', 'azimuth', 'conf_disambig']):
        """
        When not all segments download successfully, delete remaining segments.
        supply a row from a custom pandas data_frame and use this info to delete the HMI image(s).
        (a pandas series is a row of a dataframe, so these are basically
        the single image results from the query)
        - you can obtain these by doing data_framge = key=keys.iloc[row_index]
        The sub_path and filename are determined from the image information
        exit_flag: 0-Successful download; 1-download error; 2-file already exists
        """
        if len(data_series.shape) != 1:
            raise RuntimeError('data_series has more than one row!')

        # initialize fname and exit_flag
        nsegs = len(segments)

        # loop through segments
        for ii in range(nsegs):
            seg = segments[ii]

            # build the filename and subdir from the series, timestamp, and wavelength information
            # image_datetime = astropy.time.Time(data_series['time'], scale='tai').datetime
            image_datetime = astropy.time.Time.strptime(
                data_series['time'], format_string="%Y.%m.%d_%H:%M:%S", scale='tai'
            ).datetime
            prefix = '_'.join(self.series.split('.'))
            # postfix = str(data_series['filter'])
            postfix = seg
            ext = 'fits'
            dir, seg_fname = io_helpers.construct_path_and_fname(base_dir, image_datetime, prefix, postfix, ext)
            fpath = dir + os.sep + seg_fname

            # delete the file
            if os.path.isfile(fpath):
                os.remove(fpath)
                print(f"{fpath} deleted successfully.")

        return


class HMI_Mrmap_latlon_720s:
    """
    Class that holds drms/JSOC specific routines for the hmi.Mrmap_latlon_720s series
    - On initialization it sets up a client that will be used to work with this data
    """

    def __init__(self, verbose=False):

        self.series = 'hmi.Mrmap_latlon_720s'

        # initialize the drms client
        self.client = drms.Client()

        # obtain ALL of the keys for this series (i believe this connects to JSOC)
        self.allkeys = self.client.keys(self.series)

        # ---------------------------------------------
        # define the series specific default variables here
        # ---------------------------------------------
        # default keys for a normal query over a range of times
        # self.default_keys = ['**ALL**', ]
        self.default_keys = ['T_REC', 'T_OBS', 'EXPTIME', 'CAMERA', 'QUALITY']

        # default filters to use for a query
        self.default_filters = ['QUALITY=0', ]
        # self.default_filters = ['CAMERA=1', 'QUALITY=0']

        # default segments for a query
        self.default_segments = ['data']

        # header items that change from "as-is" to "fits" formats served by JSOC.
        self.hdr_keys_to_delete = ['SOURCE', ]
        # self.hdr_keys_to_add = ['T-OBS', 'CAMERA', 'T-OBS-JD', 'SPC-CRFT',
        #                         'INST']
        self.hdr_keys_to_add = ['ALL', ]

        if verbose:
            print('### Initialized DRMS client for ' + self.series)

    def update_hmi_fits_header(self, infile, drms_query, verbose=False, force=False):
        """
        read an HMI fits file, update the header, save the file
        The silentfix is to avoid warnings/and or crashes when astropy encounters nans in the header values
        """

        # open file for update
        hdu_in = astropy.io.fits.open(infile, mode='update')
        # The silentfix is to avoid warnings/and or crashes when astropy encounters nans in the header values
        hdu_in.verify('silentfix')
        hdr = hdu_in[1].header

        # check to see if this file has already been converted
        if set(self.hdr_keys_to_add).issubset(hdr) and not set(self.hdr_keys_to_delete).issubset(hdr) and not force:
            if verbose:
                print("### " + infile + " looks like it has an updated header. SKIPPING.")
            return

        # get the corresponding header info from the JSOC as a drms frame
        drms_frame = self.get_drms_info_for_image(drms_query)

        # update the header info
        hdr_update = self.update_header_fields_from_drms(hdr, drms_frame,
                                                         verbose=verbose)

        # update header and close
        hdu_in[1].header = hdr_update
        hdu_in.close()


    def get_drms_info_for_image(self, init_query):
        """
        supply a single row from the result of query_time_interval(),
        obtain the full JSOC info as a pandas frame from drms
        The prime key syntax for hmi.m_720s is used here
        """
        query_string = '%s[%s_TAI]'%(self.series, init_query.time.iloc[0])
        drms_frame = self.client.query(query_string, key="**ALL**")
        # remove unwanted fields
        all_cols = set(drms_frame.keys())
        keep_cols = all_cols.difference(set(self.hdr_keys_to_delete))
        drms_frame = drms_frame.loc[:, keep_cols]

        return drms_frame

    def update_header_fields_from_drms(self, hdr, drms_frame, verbose=False):
        """
        update the fits header using info from a drms pandas frame
        This works for converting "as-is" hmi.m_720s headers to the "fits" style
        """

        # Replace all NaNs with a flag value
        if "BLANK" in drms_frame.keys():
            blank_val = drms_frame.BLANK[0]
        else:
            blank_val = -2147483648
        drms_frame.fillna(blank_val, inplace=True)

        # Delete the unwanted "as-is" protocol keys
        if self.hdr_keys_to_delete is not None:
            for key in self.hdr_keys_to_delete:
                if key in hdr:
                    hdr.remove(key)

        # setup the conversion of some drms pandas tags to "fits" tags
        hdr_pandas_keys_to_fits = {
            'DATE__OBS': 'DATE_OBS',
            'T_REC_epoch': 'TRECEPOC',
            'T_REC_step': 'TRECSTEP',
            'T_REC_unit': 'TRECUNIT'}

        # Add a history line about the conversion
        my_history = 'Updated the JSOC "as-is" ' + self.series + ' header information to "fits" using DRMS query info.'
        hdr.add_history(my_history)

        # loop over the drms keys, update the corresponding FITS header tags
        for pandas_key in drms_frame.keys():
            fits_key = pandas_key

            if pandas_key in hdr_pandas_keys_to_fits:
                fits_key = hdr_pandas_keys_to_fits[pandas_key]

            # normal behavior fits key already exists, update w/ same time.
            if fits_key in hdr:
                fits_val = hdr[fits_key]

                pandas_val = drms_frame[pandas_key][0]
                # remove line breaks from strings
                if type(pandas_val) is str:
                    pandas_val = pandas_val.replace("\n", "")
                # trap for nan ints to maintain the standard types JSOC uses in their fits files.
                if type(fits_val) is int:
                    if math.isnan(pandas_val):
                        pandas_val = blank_val
                elif pandas_key == "HISTORY":
                    hdr.add_history(pandas_val)
                elif pandas_key == "COMMENT":
                    hdr.add_comment(pandas_val)
                else:
                    pandas_val = drms_frame[pandas_key].astype(type(fits_val))[0]

                if fits_val != pandas_val:
                    if verbose:
                        print('Diff Val: ', pandas_key, fits_key, drms_frame[pandas_key][0], hdr[fits_key])
                    hdr[fits_key] = pandas_val

            # Check for adding a key that doesn't exist.
            elif fits_key in self.hdr_keys_to_add or (len(self.hdr_keys_to_add) == 1
                                                      and self.hdr_keys_to_add[0] == "ALL"):
                hdr[fits_key] = drms_frame[pandas_key][0]

            # Check if the key is missing from the file and wasn't supposed to be added.
            else:
                if verbose:
                    print('MISSING:   ', pandas_key, drms_frame[pandas_key][0])

        # Last, check for 'DATE-OBS' field and convert to MJD
        if "DATE_OBS" in hdr.keys():
            date_str = hdr["DATE-OBS"]
            date_str = date_str.replace("Z", "")
            dtime = astropy.time.Time(date_str, format='fits')
            mjd = dtime.mjd
            hdr['MJD-OBS'] = mjd

        return hdr

    def query_time_interval(self, time_range, search_cadence=None,
                            filters=None, segments=None, keys=None):
        """
        Quick function to query the JSOC for all matching magnetograms over
        a certain interval (intervals should be specified in TAI time)
        - returns the drms pandas dataframes of the keys and segments
        - if no magnetograms, len(keys) and len(segs) will be zero
        """

        # set up the default values (these should never be passed as None anyway)
        if filters is None:
            filters = self.default_filters
        if segments is None:
            segments = self.default_segments
        if keys is None:
            keys = self.default_keys

        # build the keys string
        key_str = ', '.join(keys)

        # build the Filter Query String
        if len(filters) > 0:
            filter_str = '[? ' + ' ?][? '.join(filters) + ' ?]'
        else:
            filter_str = ''

        # build the segments string
        segment_str = '{' + ', '.join(segments) + '}'

        # build the time interval string
        if search_cadence is not None:
            if type(search_cadence) is astropy.units.quantity.Quantity:
                cadence_str = '@' + get_jsoc_interval_format(search_cadence)
            else:
                cadence_str = '@' + search_cadence
        else:
            cadence_str = ""

        # keep back-compatibility, but convert time_range to a list of datetimes
        if type(time_range) is sunpy.time.timerange.TimeRange:
            time_start = time_range.start.to_datetime()
            time_end = time_range.end.to_datetime()
        else:
            time_start = time_range[0]
            time_end = time_range[1]
        time_range_str = time_start.strftime("%Y.%m.%d_%H:%M:%S") + \
            "_TAI-" + time_end.strftime("%Y.%m.%d_%H:%M:%S") + "_TAI"
        time_query_str = "[" + time_range_str + cadence_str + "]"

        # build the query
        query_string = self.series + time_query_str + filter_str + segment_str

        # query JSOC for image times
        try:
            key_frame, seg_frame = self.client.query(query_string, key=key_str, seg=segments)
        except HTTPException:
            print("There was a problem contacting the JSOC server to query image times. Trying again...\n")
            try:
                key_frame, seg_frame = self.client.query(query_string, key=key_str, seg=segments)
            except HTTPException:
                print("Still cannot contact JSOC server. Returning 'query error'.")
                return "query error"

        if len(seg_frame) == 0:
            # No results were found for this time range. generate empty Dataframe to return
            data_frame = pd.DataFrame(columns=('spacecraft', 'instrument', 'camera', 'filter', 'time', 'jd', 'url'))
            return data_frame

        # parse the results a bit
        time_strings, jds = parse_query_times(key_frame, time_type='tai')
        # Generalize for more segments
        urls = jsoc_url + seg_frame['data']

        # make the custom dataframe with the info i want
        # data_frame = io_helpers.custom_dataframe(time_strings, jds, urls, 'SDO', 'HMI')
        data_frame = pd.DataFrame(
            OrderedDict({'spacecraft': 'SDO', 'instrument': 'HMI', 'camera': key_frame.CAMERA,
                         'filters': filter_str, 'time': time_strings, 'jd': jds, 'url': urls})
        )

        # return key_frame, seg_frame
        return data_frame

    def download_image_fixed_format(self, data_series, base_dir, update=True, overwrite=False, verbose=False):
        """
        supply a row from a custom pandas data_frame and use this info to download the HMI image.
        (a pandas series is a row of a dataframe, so these are basically
        the single image results from the query)
        - you can obtain these by doing data_framge = key=keys.iloc[row_index]
        The sub_path and filename are determined from the image information
        exit_flag: 0-Successful download; 1-download error; 2-file already exists
        """
        if len(data_series.shape) != 1:
            raise RuntimeError('data_series has more than one row!')

        # build the url
        url = data_series['url']

        # build the filename and subdir from the series, timestamp, and wavelength information
        # image_datetime = astropy.time.Time(data_series['time'], scale='tai').datetime
        image_datetime = astropy.time.Time.strptime(
            data_series['time'], format_string="%Y.%m.%d_%H:%M:%S", scale='tai'
        ).datetime
        prefix = '_'.join(self.series.split('.'))
        # postfix = str(data_series['filter'])
        postfix = ''
        ext = 'fits'
        dir, fname = io_helpers.construct_path_and_fname(base_dir, image_datetime, prefix, postfix, ext)
        fpath = dir + os.sep + fname

        # download the file
        exit_flag = io_helpers.download_url(url, fpath, verbose=verbose, overwrite=overwrite)

        if exit_flag == 1:
            # There was an error with download. Try again
            print(" Re-trying download....")
            exit_flag = io_helpers.download_url(url, fpath, verbose=verbose, overwrite=overwrite)
            if exit_flag == 1:
                # Download failed. Return None
                return None, None, exit_flag

        # update the header info if desired
        if update:
            if verbose:
                print('  Updating header info if necessary.')
            drms_query = data_series.to_frame().T
            self.update_hmi_fits_header(fpath, drms_query, verbose=False)

        # now separate the the sub directory from the base path (i want relative path for the DB)
        sub_dir = os.path.sep.join(dir.split(base_dir)[1].split(os.path.sep)[1:])

        return sub_dir, fname, exit_flag


def get_jsoc_interval_format(time_interval):
    """
    Return a string with the jsoc style way of specifying a time interval (e.g. 12s, 2h, 2d)
    - the input time interval a length of time specified as an astropy unit type (i.e. 12 * u.second).
    """
    secs = time_interval.to(astropy.units.second).value
    if secs < 60.:
        return str(secs) + 's'
    elif secs < 3600.:
        return str(time_interval.to(astropy.units.min).value) + 'm'
    elif secs < 86400.0:
        return str(time_interval.to(astropy.units.hour).value) + 'h'
    else:
        return str(time_interval.to(astropy.units.day).value) + 'd'


def build_time_string_from_range(time_range, image_search_cadence):
    """
    Quick function to build a jsoc time string from a sunpy TimeRange class
    """
    interval_str = '/' + get_jsoc_interval_format(time_range.seconds)
    cadence_str = '@' + get_jsoc_interval_format(image_search_cadence)
    time_start = astropy.time.Time(time_range.start, format='datetime', scale='utc')
    date_start = time_start.isot + time_zone_postfix
    return '%s%s%s'%(date_start, interval_str, cadence_str)


def parse_query_times(key_frame, time_type='utc'):
    """
    parse a drms keys frame to return arrays of
    - the times in string format
    - the times in jd format
    """
    if len(key_frame) > 0:
        time_strings = key_frame['T_OBS'].to_numpy().astype('str')
        if time_type == 'utc':
            time_array = astropy.time.Time(time_strings, scale='utc')
        elif time_type == 'tai':
            time_strings = np.char.replace(time_strings, "_TAI", "")
            # this 'tai' format is hmi720s-specific
            time_array = astropy.time.Time.strptime(
                time_strings, format_string="%Y.%m.%d_%H:%M:%S", scale='tai')
        jds = time_array.jd
    else:
        # type the null arrays the same as if they were full in case this can help avoid headaches later
        time_strings = np.asarray([], dtype=np.dtype('<U23'))
        jds = np.asarray([], dtype='float64')

    return time_strings, jds


def drms_query_with_retry(client, query_string, keys, max_retries=5, delay=2, **kwargs):
    """
    Execute a DRMS query with retry logic on timeout or connection failure.

    Parameters:
        client (drms.Client): The DRMS client.
        query_string (str): Data series name (e.g., "hmi.M_720s") and basic query.
        keys (str or list): Query parameters.
        max_retries (int): Number of retry attempts.
        delay (int): Seconds to wait between retries.
        **kwargs: Additional arguments for drms.Client.query().

    Returns:
        pandas.DataFrame or None: Query result if successful, None if failed.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            print(f"Attempt {attempt + 1}: DRMS query {query_string}")

            # Perform the DRMS query
            result = client.query(query_string, key=keys, **kwargs)

            print("Query successful.")
            return result  # Return the successful result

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"Network error: {e}. Retrying in {delay} seconds...")

        except drms.DrmsExportError as e:
            print(f"DRMS Export Error: {e}. Retrying in {delay} seconds...")

        except drms.DrmsQueryError as e:
            print(f"DRMS Query Error: {e}. Retrying in {delay} seconds...")

        # Generic exception handling (use cautiously)
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying in {delay} seconds...")

        attempt += 1
        time.sleep(delay)

    print("Max retries reached. Query failed.")
    return None  # Return None if the query never succeeds

# if __name__ == "__main__":
#     s12 = S12(verbose=True)
#     print('  series: ' + s12.series)
#     print('  default keys:', s12.default_keys)
#     print('  default filters:', s12.default_filters)
#     print('  default segments:', s12.default_segments)
