import rasterio

from potential.spatial import process_climate, get_metadata


class ProcessClimateData:

    def __init__(self, nc_template, varname="sfcWind", epsg="4326"):

        # get a rasterio metadata dictionary from the template NetCDF dataset
        self.metadata = get_metadata(nc_template, varname=varname, epsg=epsg)

    def to_raster(self, nc_file, output_raster, varname):
        """Write yearly mean raster to file."""

        # convert daily to yearly mean array from input netcdf
        arr = process_climate(nc_file, variable=varname)

        with rasterio.open(output_raster, "w", **self.metadata) as dst:
            dst.write_band(1, arr



