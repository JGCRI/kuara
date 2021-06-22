
import numpy as np
import rasterio

from kuara.spatial import reclassify


def calc_total_suitable_area(elev_raster, slope_raster, prot_raster, perm_raster, lulc_raster, output_directory):

    # create exclusion rasters for each exclusion category
    elev = process_elevation(elev_raster)
    slope = process_slope(slope_raster)
    prot = process_protected(prot_raster)
    perm = process_permafrost(perm_raster)
    lulc = process_lulc(lulc_raster)
    gridcellarea = np.load(pkg_resources.resource_filename('kuara', 'data/gridcell_area_km2_0p5deg.npy'))

    # replacing negative gridcell area values (-999.9) in ocean cells by 0.0
    gridcellarea = np.where(gridcellarea > 0.0, gridcellarea, 0.0)
    f_suit = calc_final_suitability(elev, slope, prot, perm, lulc)

    # calculate the suitable area in sqkm per gridcell (fi * ai)
    f_suit_sqkm = f_suit * gridcellarea[::-1, :]  # need to invert the lat index in gridcellarea

    out = os.path.join(output_directory, 'gridcellarea0p5deg.npy')
    np.save(out, gridcellarea)

    out = os.path.join(output_directory, 'wind_suitability.npy')
    np.save(out, f_suit)

    out_suit_sqkm = os.path.join(output_directory, 'wind_suitable_sqkm.npy')
    np.save(out_suit_sqkm, f_suit_sqkm)

    return f_suit_sqkm

