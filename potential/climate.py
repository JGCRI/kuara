import os

import numpy as np
import rasterio

from potential.spatial import process_climate, get_metadata


class ProcessClimateData:

    def __init__(self, raster_dir, nc_template, target_year, model,
                 wind_ncdf, pressure_ncdf, tas_ncdf, huss_ncdf,
                 wind_var="sfcWind", pressure_var="ps",
                 tas_var="tas", huss_var="huss", epsg="4326"):

        # get a rasterio metadata dictionary from the template NetCDF dataset
        self.metadata = get_metadata(nc_template, varname=wind_var, epsg=epsg)

        # input netcdf files and varaible names
        self.wind_ncdf = wind_ncdf
        self.wind_var = wind_var
        self.pressure_ncdf = pressure_ncdf
        self.pressure_var = pressure_var
        self.tas_ncdf = tas_ncdf
        self.tas_var = tas_var
        self.huss_ncdf = huss_ncdf
        self.huss_var = huss_var

        # output raster file names
        self.wind_raster_file = os.path.join(raster_dir, f"{wind_var}_{model}_{target_year}.tif")
        self.pressure_raster_file = os.path.join(raster_dir, f"{pressure_var}_{model}_{target_year}.tif")
        self.tas_raster_file = os.path.join(raster_dir, f"{tas_var}_{model}_{target_year}.tif")
        self.huss_raster_file = os.path.join(raster_dir, f"{huss_var}_{model}_{target_year}.tif")

    def to_raster(self, nc_file, output_raster, varname):
        """Write yearly mean raster to file."""

        # convert daily to yearly mean array from input netcdf
        arr = process_climate(nc_file, variable=varname)

        with rasterio.open(output_raster, "w", **self.metadata) as dst:
            dst.write_band(1, arr)

    def generate_climate_rasters(self):
        """Convert NetCDF to yearly mean rasters for each climate dataset."""

        self.to_raster(self.wind_ncdf, self.wind_raster_file, self.wind_var)
        self.to_raster(self.pressure_ncdf, self.pressure_raster_file, self.pressure_var)
        self.to_raster(self.tas_ncdf, self.tas_raster_file, self.tas_var)
        self.to_raster(self.huss_ncdf, self.huss_raster_file, self.huss_var)

    def build_mask(self):
        """Create climate land mask replacing existing NaN with np.nan and all other elements with 1
        representing area where climate data exist on land.

        :return:                    NumPy array where land == 1 and all other np.nan

        """

        with rasterio.open(self.wind_raster_file) as src:
            return np.where(np.isnan(src.read(1)), np.nan, 1)


