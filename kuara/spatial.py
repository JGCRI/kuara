import numpy as np
import rasterio


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
