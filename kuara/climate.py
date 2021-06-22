import os
import logging

import rasterio
import numpy as np
import xarray as xr

from rasterio.crs import CRS

from kuara.logger import Logger


class ProcessClimateData:
    """Process climate data NetCDF files into a 2D array for the target year.

    Output array values include the following substitutions for NaNs which are assumed to be located in ocean
    grid cells:

    near_surface_wind_speed:  NaN to 0.0
    surface_air_pressure: NaN to 0.0
    air_temperature: NaN to -999.0
    specific_humidity: NaN to 0.0

    :param target_year:                                 Four digit year to process in YYYY format
    :type target_year:                                  int

    :param near_surface_wind_speed_ncdf:                Full path with file name and extension to the near surface wind
                                                        speed NetCDF file
                                                        Units:  m s-1
    :type near_surface_wind_speed_ncdf:                 str

    :param near_surface_wind_speed_varname:             Variable name for wind in the NetCDF file
    :type near_surface_wind_speed_varname:              str

    :param surface_air_pressure_ncdf:                   Full path with file name and extension to the surface air
                                                        pressure NetCDF file
                                                        Units: Pa
    :type surface_air_pressure_ncdf:                    str

    :param surface_air_pressure_varname:                Variable name for air pressure in the NetCDF file
    :type surface_air_pressure_varname:                 str

    :param air_temperature_ncdf:                        Full path with file name and extension to the air temperature
                                                        NetCDF file
                                                        Units: K
    :type air_temperature_ncdf:                         str

    :param air_temperature_varname:                     Variable name for air temperature in the NetCDF file
    :type air_temperature_varname:                      str

    :param specific_humidity_ncdf:                      Full path with file name and extension to the near surface
                                                        specific humidity NetCDF file
                                                        Units:  unitless fraction
    :type specific_humidity_ncdf:                       str

    :param specific_humidity_varname:                   Variable name for specific humidity in the NetCDF file
    :type specific_humidity_varname:                    str

    """

    def __init__(self, target_year, near_surface_wind_speed_ncdf, near_surface_wind_speed_varname,
                 surface_air_pressure_ncdf, surface_air_pressure_varname, air_temperature_ncdf,
                 air_temperature_varname, specific_humidity_ncdf, specific_humidity_varname):

        # input netcdf files and variable names
        self.near_surface_wind_speed_ncdf = near_surface_wind_speed_ncdf
        self.near_surface_wind_speed_varname = near_surface_wind_speed_varname
        self.surface_air_pressure_ncdf = surface_air_pressure_ncdf
        self.surface_air_pressure_varname = surface_air_pressure_varname
        self.air_temperature_ncdf = air_temperature_ncdf
        self.air_temperature_varname = air_temperature_varname
        self.specific_humidity_ncdf = specific_humidity_ncdf
        self.specific_humidity_varname = specific_humidity_varname
        self.target_year = target_year

        # convert netcdf files to an array for the target year
        logging.info(f"Generating near surface wind speed array with NaN replaced with 0.0")
        self.near_surface_wind_speed_arr = self.ncdf_to_array(near_surface_wind_speed_ncdf, near_surface_wind_speed_varname, nan_replace_value=0.0)

        logging.info(f"Generating surface air pressure array with NaN replaced with 0.0")
        self.surface_air_pressure_arr = self.ncdf_to_array(surface_air_pressure_ncdf, surface_air_pressure_varname, nan_replace_value=0.0)

        logging.info(f"Generating air temperature array with NaN replaced with -999.0")
        self.air_temperature_arr = self.ncdf_to_array(air_temperature_ncdf, air_temperature_varname, nan_replace_value=-999.0)

        logging.info(f"Generating specific humidity array with NaN replaced with 0.0")
        self.specific_humidity_arr = self.ncdf_to_array(specific_humidity_ncdf, specific_humidity_varname, nan_replace_value=0.0)

    def ncdf_to_array(self, nc_file, variable, nan_replace_value):
        """Read in NetCDF file and get the mean of the desired time period as an array.

        :param nc_file:  Full path with file name and extension to the input NetCDF file
        :type nc_file: str

        :param variable:  Name of the target variable in the NetCDF file
        :type variable: str

        :return:   2D array of values for [lat, lon]


        """
        yr = str(self.target_year)

        # read in NetCDF file
        ds = xr.open_dataset(nc_file)

        # get desired time slice from data
        dsx = ds.sel(time=slice(yr, yr))

        # extract daily data for target_year
        arr = dsx.variables[variable][:, :, :].values

        # give NaN elements which correspond to ocean a value
        return np.nan_to_num(arr, nan=nan_replace_value, copy=True)

    def array_to_raster(self, nc_file, variable, output_raster, metadata):
        """Write an array to raster file."""

        arr_mean = self.nc_daily_to_monthly_mean(nc_file, variable)

        with rasterio.open(output_raster, "w", **metadata) as dst:
            dst.write_band(1, arr_mean)

    def write_rasters(self, output_dir, scenario_name='', native_epsg='4326'):
        """Write the processed climate arrays to raster files.

        :param output_dir:                                  Full path to the output directory
        :type output_dir:                                   str

        :param scenario_name:                               A desired scenario name to distinguish the output file name
                                                            from others
                                                            Default: empty string
        :type scenario_name:                                str

        :param native_epsg:                                 EPSG number of the coordinate reference system for the input
                                                            NetCDF files
                                                            Default: 4326 which is WGS84
        :type native_epsg:                                  str

        """

        # get a rasterio metadata dictionary from the template NetCDF dataset
        metadata = self.get_metadata(native_epsg=native_epsg)

        # output raster file names
        wind_raster_file = os.path.join(output_dir, f"{self.near_surface_wind_speed_varname}_{scenario_name}_{self.target_year}_mean.tif")
        pressure_raster_file = os.path.join(output_dir, f"{self.surface_air_pressure_varname}_{scenario_name}_{self.target_year}_mean.tif")
        tas_raster_file = os.path.join(output_dir, f"{self.air_temperature_varname}_{scenario_name}_{self.target_year}_mean.tif")
        huss_raster_file = os.path.join(output_dir, f"{self.specific_humidity_varname}_{scenario_name}_{self.target_year}_mean.tif")

        # write to rasters
        logging.info(f"Writing near surface wind speed data as yearly mean to raster file: {wind_raster_file}")
        self.array_to_raster(self.near_surface_wind_speed_ncdf,
                             self.near_surface_wind_speed_varname,
                             wind_raster_file,
                             metadata=metadata)

        logging.info(f"Writing surface air pressure data as yearly mean to raster file: {pressure_raster_file}")
        self.array_to_raster(self.surface_air_pressure_ncdf,
                             self.surface_air_pressure_varname,
                             pressure_raster_file,
                             metadata=metadata)

        logging.info(f"Writing air temperature data as yearly mean to raster file: {tas_raster_file}")
        self.array_to_raster(self.air_temperature_ncdf,
                             self.air_temperature_varname,
                             tas_raster_file,
                             metadata=metadata)

        logging.info(f"Writing specific humidity data as yearly mean to raster file: {huss_raster_file}")
        self.array_to_raster(self.specific_humidity_ncdf,
                             self.specific_humidity_varname,
                             huss_raster_file,
                             metadata=metadata)

    def write_npy_files(self, output_dir, scenario_name=''):
        """Write the processed climate arrays to NPY files.

        :param output_dir:                                  Full path to the output directory
        :type output_dir:                                   str

        :param scenario_name:                               A desired scenario name to distinguish the output file name
                                                            from others
                                                            Default: empty string
        :type scenario_name:                                str

        """

        # output NPY file names
        wind_npy_file = os.path.join(output_dir, f"{self.near_surface_wind_speed_varname}_{scenario_name}_{self.target_year}.npy")
        pressure_npy_file = os.path.join(output_dir, f"{self.surface_air_pressure_varname}_{scenario_name}_{self.target_year}.npy")
        tas_npy_file = os.path.join(output_dir, f"{self.air_temperature_varname}_{scenario_name}_{self.target_year}.npy")
        huss_npy_file = os.path.join(output_dir, f"{self.specific_humidity_varname}_{scenario_name}_{self.target_year}.npy")

        # write to NPY files
        logging.info(f"Writing near surface wind speed data to NPY file: {wind_npy_file}")
        np.save(wind_npy_file, self.near_surface_wind_speed_arr)

        logging.info(f"Writing surface air pressure data to NPY file: {pressure_npy_file}")
        np.save(pressure_npy_file, self.surface_air_pressure_arr)

        logging.info(f"Writing air temperature data to NPY file: {tas_npy_file}")
        np.save(tas_npy_file, self.air_temperature_arr)

        logging.info(f"Writing specific humidity data to NPY file: {huss_npy_file}")
        np.save(huss_npy_file, self.specific_humidity_arr)

    def build_mask(self, raster_file):
        """Create climate land mask replacing existing NaN with np.nan and all other elements with 1
        representing area where climate data exist on land.

        :return:                    NumPy array where land == 1 and all other np.nan

        """

        with rasterio.open(raster_file) as src:
            return np.where(np.isnan(src.read(1)), np.nan, 1)

    def get_metadata(self, native_epsg='4326'):
        """Get the rasterio metadata dictionary from a source NetCDF4 file and update with a
        GeoTIFF driver and target coordinate reference system.

        :param nc_file:                     Full path with file name and extension to the input NetCDF file
        :type nc_file:                      str

        :param variable:                    NetCDF variable name to use
        :type variable:                     str

        :param epsg:                        EPSG number of the coordinate reference system
        :type epsg:                         str

        :return:                            rasterio metadata dictionary

        """

        # create rasterio recognizable name to read netcdf file into a rasterio object
        nc_rasterio = f"netcdf:{self.near_surface_wind_speed_ncdf}:{self.near_surface_wind_speed_varname}"

        # extract metadata from input file
        with rasterio.open(nc_rasterio) as src:
            metadata = src.meta.copy()

        # update metadata
        metadata.update({"driver": "GTiff", "crs": CRS({"init": f"epsg:{native_epsg}"}), "count": 1})

        return metadata

    @staticmethod
    def nc_daily_to_monthly_mean(nc_file, variable):
        """Read in climate data to an array and convert daily data to monthly mean.
         Extract a specific year for the target variable.

        :param nc_file:                     Full path with file name and extension to the input NetCDF file
        :type nc_file:                      str

        :param variable:                    Variable name to extract
        :type variable:                     str

        :return:                            ndarray

        """

        # read in NetCDF file
        ds = xr.open_dataset(nc_file)

        # convert daily to monthly mean
        dsr = ds.resample(time='Y').mean('time')

        # extract specific year
        arr = dsr.variables[variable][0, :, :].values

        return arr


def process_climate_data(target_year, near_surface_wind_speed_ncdf, near_surface_wind_speed_varname,
                         surface_air_pressure_ncdf, surface_air_pressure_varname, air_temperature_ncdf,
                         air_temperature_varname, specific_humidity_ncdf, specific_humidity_varname, native_epsg="4326",
                         write_rasters=False, write_npy=False, output_dir='', scenario_name=''):
    """Convenience wrapper for ProcessClimateData.  Process climate data NetCDF files into a 2D array for the target
    year.

    Output array values include the following substitutions for NaNs which are assumed to be located in ocean
    grid cells:

    near_surface_wind_speed:  NaN to 0.0
    surface_air_pressure: NaN to 0.0
    air_temperature: NaN to -999.0
    specific_humidity: NaN to 0.0

    :param target_year:                                 Four digit year to process in YYYY format
    :type target_year:                                  int

    :param near_surface_wind_speed_ncdf:                Full path with file name and extension to the near surface wind
                                                        speed NetCDF file
                                                        Units:  m s-1
    :type near_surface_wind_speed_ncdf:                 str

    :param near_surface_wind_speed_varname:             Variable name for wind in the NetCDF file
    :type near_surface_wind_speed_varname:              str

    :param surface_air_pressure_ncdf:                   Full path with file name and extension to the surface air
                                                        pressure NetCDF file
                                                        Units: Pa
    :type surface_air_pressure_ncdf:                    str

    :param surface_air_pressure_varname:                Variable name for air pressure in the NetCDF file
    :type surface_air_pressure_varname:                 str

    :param air_temperature_ncdf:                        Full path with file name and extension to the air temperature
                                                        NetCDF file
                                                        Units: K
    :type air_temperature_ncdf:                         str

    :param air_temperature_varname:                     Variable name for air temperature in the NetCDF file
    :type air_temperature_varname:                      str

    :param specific_humidity_ncdf:                      Full path with file name and extension to the near surface
                                                        specific humidity NetCDF file
                                                        Units:  unitless fraction
    :type specific_humidity_ncdf:                       str

    :param specific_humidity_varname:                   Variable name for specific humidity in the NetCDF file
    :type specific_humidity_varname:                    str

    :param native_epsg:                                 EPSG number of the coordinate reference system for the input
                                                        NetCDF files
                                                        Default: 4326 which is WGS84
    :type native_epsg:                                  str

    :param write_rasters:                               Choose to write processed climate data as a raster
                                                        Default: False
    :type write_rasters:                                bool

    :param write_npy:                                   Choose to write processed climate data as a NPY file
                                                        Default: False
    :type write_npy:                                    bool

    :param output_dir:                                  If writing to file, specify an output directory
                                                        Default: empty string
    :type output_dir:                                   str

    :param scenario_name:                               If writing to file, a desired scenario name to distinguish the
                                                        output file name from others
                                                        Default: empty string
    :type scenario_name:                                str

    :returns:                                           An object containing the following processed arrays:
                                                        near_surface_wind_speed_arr
                                                        surface_air_pressure_arr
                                                        air_temperature_arr
                                                        specific_humidity_arr

    """

    # initialize logger with format and handler
    log = Logger()

    logging.info(f"Processing climate data for year:  {target_year}")

    try:

        # generate climate data
        climate = ProcessClimateData(target_year=target_year,
                                     near_surface_wind_speed_ncdf=near_surface_wind_speed_ncdf,
                                     surface_air_pressure_ncdf=surface_air_pressure_ncdf,
                                     air_temperature_ncdf=air_temperature_ncdf,
                                     specific_humidity_ncdf=specific_humidity_ncdf,
                                     near_surface_wind_speed_varname=near_surface_wind_speed_varname,
                                     surface_air_pressure_varname=surface_air_pressure_varname,
                                     air_temperature_varname=air_temperature_varname,
                                     specific_humidity_varname=specific_humidity_varname)

        # make scenario name lower case separated by hyphens where spaces exists
        scenario_name = '-'.join(scenario_name.split()).lower()

        if (write_rasters or write_npy) and (output_dir == ''):

            msg = "Must provide value for 'output_dir' if writing to file."

            logging.error(msg)
            raise NotADirectoryError(msg)

        if write_rasters:
            climate.write_rasters(output_dir, scenario_name=scenario_name, native_epsg=native_epsg)

        if write_npy:
            climate.write_npy_files(output_dir, scenario_name=scenario_name)

    finally:

        logging.info(f"Completed processing climate data for year:  {target_year}")
        log.close_logger()

    return climate
