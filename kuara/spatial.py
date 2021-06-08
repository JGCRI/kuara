import numpy as np
import rasterio
import xarray as xr

from rasterio.crs import CRS


def process_climate(nc_file, variable):
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


def get_metadata(nc_file, varname, epsg="4326"):
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
    nc_rasterio = f"netcdf:{nc_file}:{varname}"

    # extract metadata from input file
    with rasterio.open(nc_rasterio) as src:
        metadata = src.meta.copy()

    # update metadata
    metadata.update({"driver": "GTiff", "crs": CRS({"init": f"epsg:{epsg}"}), "count": 1})

    return metadata


def reclassify(raster_file, threshold):
    """Calculate suitability as > threshold == 0, else 1.

    :param raster_file:                     Full path with file name and extension to the input raster file
    :type raster_file:                      str

    :param threshold:                       Number at which above will be suitable in map units
    :type threshold:                        int

    :return:                                NumPy array of 0, 1 where 0 is suitable

    """

    with rasterio.open(raster_file) as src:
        return np.where(src.read(1) > threshold, 0, 1)
