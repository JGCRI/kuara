import os
import calendar
import pkg_resources

import numpy as np
import pandas as pd
from scipy import interpolate
import xarray as xr
import rasterio


def extrap_wind_speed_at_hub_height(wind_speed: np.ndarray,
                                    hub_height_m: float = 100.0,
                                    ref_height_m: float = 10.0,
                                    wind_power_law_exp: float = (1.0 / 7.0)) -> np.ndarray:

    """
    Extrapolates wind speed of a reference height to the turbine hub height by using a power law.

    Sources: Karnauskas et al. 2018, https://doi.org/10.1038/s41561-017-0029-9
             Hsu et al., 1994, https://doi.org/10.1175/1520-0450(1994)033<0757:DTPLWP>2.0.CO;2

    Parameters:
        :param wind_speed:                          wind speed (m/s) at ref. height
        :type wind_speed:                           numpy array

        :param hub_height_m:                        turbine hub height (m), default 100m
        :type hub_height_m:                         float

        :param ref_height_m:                        ref. height (m) of wind speed, default 10m
        :type ref_height_m:                         float

        :param wind_power_law_exp:                  exponent of wind power law, default 1/7 for onshore wind,
                                                    1/9 for offshore wind
        :type wind_power_law_exp:                   float

        :return wind_speed_hub_ht:                  array of wind speed (m/s) at the turbine hub height
    """

    wind_speed_hub_ht = wind_speed * ((hub_height_m / ref_height_m) ** wind_power_law_exp)

    return wind_speed_hub_ht


def dry_air_density_ideal(pressure: np.ndarray,
                          temperature: np.ndarray) -> np.ndarray:

    """
    Computes dry air density based on the ideal gas law.

    Source: Karnauskas et al. 2018, https://doi.org/10.1038/s41561-017-0029-9

    Parameters:
        :param pressure:                        surface pressure (Pascal = J/m3)
        :type pressure:                         numpy array

        :param temperature:                     surface temperature (K)
        :type temperature:                      numpy array

        :return dry_air_dens:                   array of dry air density (kg/m3)
    """

    # specific gas constant of air (J / Kg * K)
    sp_gas_constant = 287.058

    dry_air_dens = pressure / (sp_gas_constant * temperature)

    return dry_air_dens


def dry_air_density_humidity(dry_air_dens: np.ndarray,
                             sp_humidity: np.ndarray) -> np.ndarray:

    """
    Corrects dry air density for humidity

    Source: Karnauskas et al. 2018, Nature Geoscience, https://doi.org/10.1038/s41561-017-0029-9

    Parameters:
        :param dry_air_dens:                    dry air density (kg/m3)
        :type dry_air_dens:                     numpy array

        :param sp_humidity:                     surface specific humidity (Kg/Kg)
        :type sp_humidity:                      numpy array

        :return dry_air_dens_hum:               array of air density (kg/m3) corrected for humidity
    """

    dry_air_dens_hum = dry_air_dens * ((1.0 + sp_humidity) / (1.0 + 1.609 * sp_humidity))

    return dry_air_dens_hum


def adjust_wind_speed_for_air_density(wind_speed: np.ndarray,
                                      dry_air_dens_hum: np.ndarray) -> np.ndarray:

    """
    Adjust wind speed to account for the differences in air density between rotor elevations
    and standard (sea level) condition.

    Sources: Karnauskas et al. 2018, https://doi.org/10.1038/s41561-017-0029-9; IEC61400-12-1 (2005)

    Parameters:
        :param wind_speed:                      wind speed (m/s)
        :type wind_speed:                       numpy array

        :param dry_air_dens_hum:                air density (kg/m3) corrected for humidity
        :type dry_air_dens_hum:                 numpy array

        :return wind_speed_adj:                 array of wind speed (m/s) adjusted for air density
    """

    wind_speed_adj = wind_speed * ((dry_air_dens_hum / 1.225) ** (1.0 / 3.0))

    return wind_speed_adj


def common_wind_power_curve(wind_speed_hub_ht_adj: np.ndarray,
                            wind_to_fit: np.ndarray,
                            power_to_fit: np.ndarray,
                            min_wind_speed: float = 2.0,
                            max_wind_speed: float = 25.0) -> np.ndarray:

    """
    Common function to compute wind power for most of the turbine types

    Parameters:
        :param wind_speed_hub_ht_adj:           wind speeds (m/s) at the turbine hub height, adjusted for air density
        :type wind_speed_hub_ht_adj:            numpy array

        :param wind_to_fit:                     turbine-specific wind speed (m/s) data points to fit power curve
        :type wind_to_fit:                      numpy array

        :param power_to_fit:                    turbine-specific power (kW) data points to fit power curve
        :type power_to_fit:                     numpy array

        :param min_wind_speed:                  turbine-specific minimum wind speed (m/s) limit, default 2.0 m/s
        :type min_wind_speed:                   float

        :param max_wind_speed:                  turbine-specific maximum wind speed (m/s) limit, default 25.0 m/s
        :type max_wind_speed:                   float

        :return power_interp:                   array of wind power (kW) for specified turbine type
    """

    # filtering wind speed input data for the min and max speed range of a turbine type
    idx_wind_filt = np.where((wind_speed_hub_ht_adj <= max_wind_speed) * (wind_speed_hub_ht_adj >= min_wind_speed))
    wind_filt = wind_speed_hub_ht_adj[idx_wind_filt]

    # linear interpolation: (x, y) pair
    f = interpolate.interp1d(wind_to_fit, power_to_fit, fill_value="extrapolate")

    # calculating wind power as a linear interpolation between points in the power curve
    power_interp = f(wind_filt)

    return power_interp


def compute_wind_power(wind_speed_hub_ht_adj: np.ndarray,
                       wind_turbname: str) -> np.ndarray:

    """
    Compute wind power for a specified turbine type

    Sources of turbine specifications:
        GE1500:             https://www.en.wind-turbine-models.com/turbines/565-general-electric-ge-1.5s

        GE_2500:            https://www.ge.com/in/wind-energy/2.5-MW-wind-turbine
                            https://www.thewindpower.net/turbine_en_382_ge-energy_2.5-100.php
                            (commissioning: 2006, indicated for low and medium wind conditions)

        vestas_v136_3450:   https://www.vestas.com/en/products/4-mw-platform/v136-_3_45_mw#!about
                            https://www.thewindpower.net/turbine_en_1074_vestas_v136-3450.php
                            (commissioning: 2015, indicated for low and medium wind conditions)

        vestas_v90_2000:    https://www.vestas.com/en/products/4-mw-platform/v136-_3_45_mw#!about
                            https://www.thewindpower.net/turbine_en_32_vestas_v90-2000.php
                            (commissioning: 2004, indicated for low and medium wind conditions)

        E101_3050:          https://www.thewindpower.net/turbine_en_924_enercon_e101-3050.php
                            https://www.enercon.de/fileadmin/Redakteur/Medien-Portal/broschueren/pdf/en/ENERCON_Produkt_en_06_2015.pdf
                            (commissioning: 2012, indicated for medium wind conditions)

        Gamesa_G114_2000:   https://en.wind-turbine-models.com/turbines/428-gamesa-g114-2.0mw
                            https://www.thewindpower.net/turbine_en_860_gamesa_g114-2000.php
                            (indicated for low wind conditions)

        IEC_classII_3500:   Eurek et al. 2017,  http://dx.doi.org/10.1016/j.eneco.2016.11.015

    Parameters:
        :param wind_speed_hub_ht_adj:           wind speeds (m/s) at the turbine hub height, adjusted for air density
        :type wind_speed_hub_ht_adj:            numpy array

        :param wind_turbname:                   wind turbine name
        :type wind_turbname:                    str

        :return power_arr:                      array of wind power (kW) for specified turbine type
    """

    # import wind power data
    wind_power_file = pkg_resources.resource_filename('kuara', 'data/wind_power_data_points_to_fit.csv')
    df_wind_power_to_fit = pd.read_csv(wind_power_file, header=0)

    # Wind points of the wind power curve
    wind_to_fit = np.array(df_wind_power_to_fit['wind_' + wind_turbname])

    # Wind power points of the wind power curve
    power_to_fit = np.array(df_wind_power_to_fit['power_' + wind_turbname])

    # Exclude nan if any
    wind_to_fit = wind_to_fit[~np.isnan(wind_to_fit)]
    power_to_fit = power_to_fit[~np.isnan(power_to_fit)]

    wind_speed_range = {'GE1500': (3.5, 25.0),
                        'GE_2500': (3.0, 25.0),
                        'vestas_v136_3450': (2.5, 22.0),
                        'vestas_v90_2000': (3.0, 25.0),
                        'E101_3050': (2.0, 25.0),
                        'Gamesa_G114_2000': (2.5, 25.0),
                        'IEC_classII_3500': (4.0, 25.0)
                        }

    if wind_turbname == 'GE1500':

        # Generate Sixth Order Fit Coef
        coef_lin = np.polyfit(wind_to_fit, power_to_fit, 6)

        # Generate Sixth Order Fit Formula
        lfit = np.poly1d(coef_lin)

        # Aplying to the wind dataset
        res = np.ma.where((wind_speed_hub_ht_adj < 13.5) * (wind_speed_hub_ht_adj > wind_speed_range[wind_turbname][0]))
        power_arr = lfit(wind_speed_hub_ht_adj[res])

        power_arr = np.where(
            (wind_speed_hub_ht_adj <= wind_speed_range[wind_turbname][1]) * (wind_speed_hub_ht_adj >= 13.5), 1500.0,
            power_arr)


    elif wind_turbname == 'IEC_classII_3500':
        power_arr = common_wind_power_curve(wind_speed_hub_ht_adj,
                                            wind_to_fit,
                                            power_to_fit,
                                            min_wind_speed=wind_speed_range[wind_turbname][0],
                                            max_wind_speed=15.0)

        # Filling out the power_arr for wind speed in the 15.0 - 25.0 m/s range
        power_arr = np.where(
            (wind_speed_hub_ht_adj <= wind_speed_range[wind_turbname][1]) * (wind_speed_hub_ht_adj >= 15.0), 3500.0,
            power_arr)


    else:
        power_arr = common_wind_power_curve(wind_speed_hub_ht_adj,
                                            wind_to_fit,
                                            power_to_fit,
                                            min_wind_speed=wind_speed_range[wind_turbname][0],
                                            max_wind_speed=wind_speed_range[wind_turbname][1])

    return power_arr


def adjust_pv_panel_eff_for_atm_condition(temp_ambient_k: np.ndarray,
                                          radiation: np.ndarray,
                                          wind_speed: np.ndarray,
                                          standard_panel_eff: float = 0.17,
                                          temp_ref_c: float = 25.0,
                                          eff_response_coef: float = -0.005,
                                          thermal_coef1: float = 4.3,
                                          thermal_coef2: float = 0.943,
                                          thermal_coef3: float = 0.028,
                                          thermal_coef4: float = -1.528) -> np.ndarray:

    """
    Computes PV panel efficiency, adjusted for atmospheric conditions.

    Source: Gernaat et al. 2021; https://doi.org/10.1038/s41558-020-00949-9

    Parameters:
        :param temp_ambient_k:                  surface air temperature (K)
        :type temp_ambient_k:                   numpy array

        :param radiation:                       solar radiation (W/m^2)
        :type radiation:                        numpy array

        :param wind_speed:                      wind speeds (m/s) at 10m height
        :type wind_speed:                       numpy array

        :param standard_panel_eff:              PV panel efficiency under standard test condition, default 17%
        :type standard_panel_eff                float

        :param temp_ref_c:                      reference temperature (deg_C) at standard condition, default 25 deg_C
        :type temp_ref_c:                       float

        :param eff_response_coef:               efficiency response of PV panel to deviation of atmospheric condition,
                                                default -0.005 (1/deg_C) is for monocrystalline silicon PV panels
        :type eff_response_coef:                float

        :param thermal_coef1:                   thermal coefficient c1, default 4.3 (deg_C)
        :type thermal_coef1:                    float

        :param thermal_coef2:                   thermal coefficient c2, default 0.943 (non-dimensional)
        :type thermal_coef2:                    float

        :param thermal_coef3:                   thermal coefficient c3, default 0.028 (deg_C * m^2 * W^-1)
        :type thermal_coef3:                    float

        :param thermal_coef4:                   thermal coefficient c4, default -1.528 (deg_C * s * m^-1)
        :type thermal_coef4:                    float

        :return pv_panel_eff_adj:               array of PV panel efficiency, adjusted for atmospheric conditions
    """

    # Convert ambient temperature from K to deg_C
    temp_ambient_c = temp_ambient_k - 273.15

    # Compute the PV panel temperature
    temp_pv_panel_c = thermal_coef1 + thermal_coef2 * temp_ambient_c + thermal_coef3 * radiation + \
                      thermal_coef4 * wind_speed

    # Compute PV panel efficiency, adjusted for atmospheric conditions
    pv_panel_eff_adj = standard_panel_eff * (1.0 + eff_response_coef * (temp_pv_panel_c - temp_ref_c))

    return pv_panel_eff_adj


def compute_full_load_hours_for_csp(radiation: np.ndarray) -> np.ndarray:

    """
    Checks the minimum feasibility threshold level for CSP operation regarding solar
    radiation and computes the full load hours (FLH) as a function of the incident solar
    radiation for a CSP plant SM 2.  Minimum taken as 3000 Wh/m2/day (or an average of
    300 W/m2 over a 10-hour solar day). This translates into 1095 kWh/m2/year.

    Sources: Koberle et al. 2015; http://dx.doi.org/10.1016/j.energy.2015.05.145
             Gernaat et al. 2021; https://doi.org/10.1038/s41558-020-00949-9

    Parameters:
        :param radiation:                       solar radiation (W/m^2)
        :type radiation:                        numpy array

        :return full_load_hours:                array of full load hours for CSP
    """

    full_load_hours = np.zeros_like(radiation)

    full_load_hours = np.where(radiation > 2800.0, 5260.0, full_load_hours)

    idx_full_load_hours = np.where((radiation >= 1095.0) * (radiation <= 2800.0))
    full_load_hours[idx_full_load_hours] = 1.83 * radiation[idx_full_load_hours] + 150.0

    full_load_hours = np.where(full_load_hours > 5260.0, 5260.0, full_load_hours)

    return full_load_hours


def compute_csp_eff(temp_ambient_k: np.ndarray,
                    radiation: np.ndarray,
                    rankine_cycle_eff: float = 0.40,
                    temp_absorber_fluid_c: float = 115.0,
                    thermal_coef_k0: float = 0.762,
                    thermal_coef_k1: float = 0.2125) -> np.ndarray:

    """
    Computes the thermal efficiency of a CSP system.

    Source: Gernaat et al. 2021; https://doi.org/10.1038/s41558-020-00949-9

    Parameters:
        :param temp_ambient_k:                  surface air temperature (K)
        :type temp_ambient_k:                   numpy array

        :param radiation:                       solar radiation (W/m^2)
        :type radiation:                        numpy array

        :param rankine_cycle_eff:               efficiency of Rankine cycle, default 40%
        :type rankine_cycle_eff:                float

        :param temp_absorber_fluid_c:           temperature of fluid in absorber, default 115 (deg_C)
        :type temp_absorber_fluid_c:            float

        :param thermal_coef_k0:                 thermal coefficient k0, default 0.762 (non-dimensional)
        :type thermal_coef_k0:                  float

        :param thermal_coef_k1:                 thermal coefficient k1, default 0.2125 (W m^−2 deg C^−1)
        :type thermal_coef_k1:                  float

        :return csp_efficiency:                 array of CSP efficiency
    """

    # Converting temp from K to deg_C
    temp_ambient_c = temp_ambient_k - 273.15

    # Initialize the array of CSP efficiency with 0s
    csp_efficiency = np.zeros_like(radiation)

    # Avoid division by zero (ocean cells and a group of land cells at higher latitudes in the Winter
    # have 0.0 values for the rad array). Setting a cutoff value of 1.0 W/m2 does not affect the
    # estimate of solar CSP potential since this value is much lower than the minimum operational
    # requirement of 300 W/m2.
    idx = np.where(radiation >= 1.0)

    # Use indexes above (idx) to compute the CSP efficiency
    csp_efficiency[idx] = rankine_cycle_eff * \
                          (thermal_coef_k0 -
                           (thermal_coef_k1 * (temp_absorber_fluid_c - temp_ambient_c[idx])) / radiation[idx])

    # Filtering negative values that can occur at higher latitudes in the Winter
    csp_efficiency = np.where(csp_efficiency <= 0.0, 0.0, csp_efficiency)

    return csp_efficiency


def read_climate_data(nc_file: str,
                      clim_varname: str,
                      target_year: int) -> np.ndarray:

    """
    Read in NetCDF file and get the daily data of desired climate variable for a target year as an array.

    Parameters:
        :param nc_file:             full path with file name and extension to the input NetCDF file
        :type nc_file:              str

        :param clim_varname:        name of the target climate variable in the NetCDF file
        :type clim_varname:         str

        :param target_year:         target year to process in YYYY format
        :type target_year:          int; str

        :return:                    n-dimensional array of climate variable with values for [time, lat, lon].
    """

    # read target_year
    yr = str(target_year)

    # read in NetCDF file
    clim_data_raw = xr.open_dataset(nc_file)

    # get desired time slice from data
    clim_data_sliced = clim_data_raw.sel(time=slice(yr, yr))

    # extract daily data for target_year
    arr_daily_clim_data = clim_data_sliced.variables[clim_varname][:, :, :].values

    return arr_daily_clim_data


def process_climate_data(radiation_nc_file: str,
                         wind_nc_file: str,
                         tas_nc_file: str,
                         ps_nc_file: str,
                         huss_nc_file: str,
                         target_year: int,
                         output_directory: str,
                         radiation_varname: str = 'rsdsAdjust',
                         wind_varname: str = 'sfcWind',
                         tas_varname: str = 'tas',
                         ps_varname: str = 'ps',
                         huss_varname: str = 'huss') -> [np.ndarray,
                                                         np.ndarray,
                                                         np.ndarray,
                                                         np.ndarray,
                                                         np.ndarray]:

    """
    Process each required climate data NetCDF file.

    Parameters:
        :param *_nc_file:                   full path with file name and extension to the input NetCDF file
        :type *_nc_file:                    str

        :param *_varname:                   name of the target climate variable in the NetCDF file
        :type *_varname:                    str

        :param target_year:                 target year to process in YYYY format
        :type target_year:                  int; str

        :param output_directory:            full path to output directory
        :type output_directory:             str

        :return:                            n-dimensional arrays of climate variables with values for [time, lat, lon]
    """

    # read daily climate data for target year from nc files
    arr_radiation = read_climate_data(radiation_nc_file, radiation_varname, target_year)
    arr_wind = read_climate_data(wind_nc_file, wind_varname, target_year)
    arr_tas = read_climate_data(tas_nc_file, tas_varname, target_year)
    arr_ps = read_climate_data(ps_nc_file, ps_varname, target_year)
    arr_huss = read_climate_data(huss_nc_file, huss_varname, target_year)

    # Replace NaN values correspond to ocean cells to avoid "RuntimeWarning"
    arr_radiation[np.isnan(arr_radiation)] = 0.0
    arr_wind[np.isnan(arr_wind)] = 0.0
    arr_tas[np.isnan(arr_tas)] = -999.0  # non-zero to avoid "RuntimeWarning"
    arr_ps[np.isnan(arr_ps)] = 0.0
    arr_huss[np.isnan(arr_huss)] = 0.0

    # calculate daily to yearly mean for radiation, wind, and tas
    arr_radiation_mean = np.mean(arr_radiation, axis=0)
    arr_wind_mean = np.mean(arr_wind, axis=0)
    arr_tas_mean = np.mean(arr_tas, axis=0)

    # save the daily mean of radiation, wind, and tas as .npy
    arr_radiation_raster = os.path.join(output_directory, 'solar_radiation_w_m2_' + str(target_year) + '.npy')
    np.save(arr_radiation_raster, arr_radiation_mean)

    arr_wind_raster = os.path.join(output_directory, 'wind_speed_ms_' + str(target_year) + '.npy')
    np.save(arr_wind_raster, arr_wind_mean)

    arr_tas_raster = os.path.join(output_directory, 'tas_deg_k_' + str(target_year) + '.npy')
    np.save(arr_tas_raster, arr_tas_mean)

    return arr_radiation, arr_wind, arr_tas, arr_ps, arr_huss


def process_elevation(elev_raster_file: str) -> np.ndarray:

    """
    Process elevation raster files to exclude unsuitable area.

    Parameters:
        :param elev_raster_file:            full path with file name and extension to the input raster file
        :type elev_raster_file:             str

        :return:                            array of elevation data
    """

    ras_elev = rasterio.open(elev_raster_file)

    arr_elev = np.where(ras_elev.read(1) > 2500, 0, 1)

    return arr_elev


def process_elevation_solar(elev_raster_file: str) -> np.ndarray:

    """
    Process elevation raster files to exclude unsuitable area for solar (no altitude constraint).

    Source: Deng et al. 2015

    Parameters:
        :param elev_raster_file:            full path with file name and extension to the input raster file
        :type elev_raster_file:             str

        :return:                            array of elevation data
    """

    ras_elev = rasterio.open(elev_raster_file)

    arr_elev_solar = np.where(ras_elev.read(1) > 2500, 1, 1)

    return arr_elev_solar


def process_slope(slope_raster_file: str) -> np.ndarray:

    """
    Process slope raster files to exclude unsuitable area.

    Parameters:
        :param slope_raster_file:           full path with file name and extension to the input raster file
        :type slope_raster_file:            str

        :return:                            array of slope data
    """

    ras_slope = rasterio.open(slope_raster_file)

    arr_slope = np.where(ras_slope.read(1) > 20, 0, 1)

    return arr_slope


def process_slope_solar_pv(slope_raster_file: str) -> np.ndarray:

    """
    Process slope raster files to exclude unsuitable area for solar PV.

    Source: Deng et al. 2015

    Parameters:
        :param slope_raster_file:           full path with file name and extension to the input raster file
        :type slope_raster_file:            str

        :return:                            array of slope data
    """

    ras_slope = rasterio.open(slope_raster_file)

    arr_slope_solar_pv = np.where(ras_slope.read(1) > 27, 0, 1)

    return arr_slope_solar_pv


def process_slope_solar_csp(slope_raster_file: str) -> np.ndarray:

    """
    Process slope raster files to exclude unsuitable area for solar CSP.

    Source: Deng et al. 2015

    Parameters:
        :param slope_raster_file:           full path with file name and extension to the input raster file
        :type slope_raster_file:            str

        :return:                            array of slope data
    """

    ras_slope = rasterio.open(slope_raster_file)

    arr_slope_solar_csp = np.where(ras_slope.read(1) > 4, 0, 1)

    return arr_slope_solar_csp


def process_protected_areas(protected_areas_raster_file: str) -> np.ndarray:

    """
    Process raster files for protected areas to exclude unsuitable area.

    Parameters:
        :param protected_areas_raster_file:         full path with file name and extension to the input raster file
        :type protected_areas_raster_file:          str

        :return:                                    array of protected areas data
    """

    ras_protected_areas = rasterio.open(protected_areas_raster_file)

    arr_protected_areas = np.where(ras_protected_areas.read(1) < 0.0, 1, 0)

    return arr_protected_areas


def process_permafrost(permafrost_raster_file: str) -> np.ndarray:

    """
    Process permafrost raster files to exclude unsuitable area.

    Parameters:
        :param permafrost_raster_file:          full path with file name and extension to the input raster file
        :type permafrost_raster_file:           str

        :return:                                array of permafrost data
    """

    ras_permafrost = rasterio.open(permafrost_raster_file)

    arr_permafrost = np.where(ras_permafrost.read(1) >= 0.1, 0, 1)

    return arr_permafrost


def process_permafrost_solar(permafrost_raster_file: str) -> np.ndarray:

    """
    Process permafrost raster files to exclude unsuitable area for solar .

    Parameters:
        :param permafrost_raster_file:          full path with file name and extension to the input raster file
        :type permafrost_raster_file:           str

        :return:                                array of permafrost data
    """

    ras_permafrost = rasterio.open(permafrost_raster_file)

    arr_permafrost_solar = np.where(ras_permafrost.read(1) >= 0.1, 1, 1)

    return arr_permafrost_solar


def process_lulc(lulc_raster_file: str) -> np.ndarray:

    """
    Process land use land cover (lulc) raster files to exclude unsuitable area.

    Source: Eurek et al. 2017

    Parameters:
        :param lulc_raster_file:                full path with file name and extension to the input raster file
        :type lulc_raster_file:                 str

        :return:                                array of lulc data
    """

    ras_lulc = rasterio.open(lulc_raster_file)

    arr_lulc = ras_lulc.read(1).astype(np.float64)

    # replace existing lulc data points with desired values
    exist = [11, 14, 20, 30, 40, 50, 60, 70, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230]
    replace = [0, 0.7, 0.7, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.65, 0.5, 0.8, 0.9, 0, 0, 0, 0, 0.9, 0, 0, 0]

    for i in range(len(exist)):
        arr_lulc = np.where(arr_lulc == exist[i], replace[i], arr_lulc)

    return arr_lulc


def process_lulc_solar(lulc_raster_file: str) -> np.ndarray:

    """
    Process land use land cover (lulc) raster files to exclude unsuitable area for solar.

    Source: Suitability factors based on Gernaat et al. 2021 and Korfiati et al. 2016

    Parameters:
        :param lulc_raster_file:                full path with file name and extension to the input raster file
        :type lulc_raster_file:                 str

        :return:                                array of lulc data
    """

    ras_lulc = rasterio.open(lulc_raster_file)

    arr_lulc_solar = ras_lulc.read(1).astype(np.float64)

    # replace existing lulc data points with desired values
    exist = [11, 14, 20, 30, 40, 50, 60, 70, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230]
    replace = [0, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0.05, 0, 0, 0]

    for i in range(len(exist)):
        arr_lulc_solar = np.where(arr_lulc_solar == exist[i], replace[i], arr_lulc_solar)

    return arr_lulc_solar


def calc_final_suitability(elev_array: np.ndarray,
                           slope_array: np.ndarray,
                           prot_array: np.ndarray,
                           perm_array: np.ndarray,
                           lulc_array: np.ndarray) -> np.ndarray:

    """
    Calculate the suitability factor per grid-cell where 0 is unsuitable and 1 is the most suitable.

    Parameters:
        :param elev_array:                array of elevation
        :type elev_array:                 numpy array

        :param slope_array:               array of slope
        :type slope_array:                numpy array

        :param prot_array:                array of protected areas
        :type prot_array:                 numpy array

        :param perm_array:                array of permafrost
        :type perm_array:                 numpy array

        :param lulc_array:                array of land use land cover (lulc)
        :type lulc_array:                 numpy array

        :return:                          array of suitability factor
    """

    arr_suitability_factor = elev_array * slope_array * prot_array * perm_array * lulc_array

    return arr_suitability_factor


def get_hours_per_year(target_year: int) -> int:

    """
    Get the hours per week for leap and non-leap years based on the target year.

    Parameters:
        :param target_year:                             Four digit year in YYYY format
        :type target_year:                              int

        :return:                                        Number of hours in a year
    """

    leap_hours = 8784
    year_hours = 8760

    if calendar.isleap(target_year):
        yr_hours = leap_hours
    else:
        yr_hours = year_hours

    return yr_hours


def calc_total_suitable_area_solar_pv(elev_raster: str,
                                      slope_raster: str,
                                      prot_area_raster: str,
                                      permafrost_raster: str,
                                      lulc_raster: str,
                                      output_directory: str,
                                      gridcell_area_raster: str) -> np.ndarray:

    """
    Calculate total suitable area for solar PV.

    Parameters:
        :param *_raster:                    full path with file name and extension to the input raster file
        :type *_raster:                     str

        :param output_directory:            full path to output directory
        :type output_directory:             str

        :return:                            array of suitable area (sqkm) for solar PV
    """

    # apply exclusion criteria for elevation, slope, protected area, permafrost, and lulc
    elev = process_elevation_solar(elev_raster)
    slope = process_slope_solar_pv(slope_raster)
    prot_area = process_protected_areas(prot_area_raster)
    permafrost = process_permafrost_solar(permafrost_raster)
    lulc = process_lulc_solar(lulc_raster)

    # load grid_cell_area raster file
    grid_cell_area = np.load(gridcell_area_raster)

    # replacing negative gridcell area values (-999.9) in ocean cells by 0.0
    grid_cell_area = np.where(grid_cell_area > 0.0, grid_cell_area, 0.0)

    # calculate suitability factor for solar PV
    suitability_factor_pv = calc_final_suitability(elev, slope, prot_area, permafrost, lulc)

    # calculate the suitable area in sqkm per grid_cell (fi * ai)
    suitable_area_pv_sqkm = suitability_factor_pv * grid_cell_area[::-1, :]

    out_grid_area = os.path.join(output_directory, 'grid_cell_area_0p5deg.npy')
    np.save(out_grid_area, grid_cell_area)

    out_suit_f_pv = os.path.join(output_directory, 'solar_pv_suitability_factor.npy')
    np.save(out_suit_f_pv, suitability_factor_pv)

    out_suit_area_pv = os.path.join(output_directory, 'solar_pv_suitable_area_sqkm.npy')
    np.save(out_suit_area_pv, suitable_area_pv_sqkm)

    return suitable_area_pv_sqkm


def calc_total_suitable_area_solar_csp(elev_raster: str,
                                       slope_raster: str,
                                       prot_area_raster: str,
                                       permafrost_raster: str,
                                       lulc_raster: str,
                                       output_directory: str,
                                       gridcell_area_raster: str) -> np.ndarray:

    """
    Calculate total suitable area for solar CSP.

    Parameters:
        :param *_raster:                    full path with file name and extension to the input raster file
        :type *_raster:                     str

        :param output_directory:            full path to output directory
        :type output_directory:             str

        :return:                            array of suitable area (sqkm) for solar CSP
    """

    # apply exclusion criteria for elevation, slope, protected area, permafrost, and lulc
    elev = process_elevation_solar(elev_raster)
    slope = process_slope_solar_csp(slope_raster)
    prot_area = process_protected_areas(prot_area_raster)
    permafrost = process_permafrost_solar(permafrost_raster)
    lulc = process_lulc_solar(lulc_raster)

    # load grid_cell_area raster file
    grid_cell_area = np.load(gridcell_area_raster)

    # replacing negative gridcell area values (-999.9) in ocean cells by 0.0
    grid_cell_area = np.where(grid_cell_area > 0.0, grid_cell_area, 0.0)

    # calculate suitability factor for solar CSP
    suitability_factor_csp = calc_final_suitability(elev, slope, prot_area, permafrost, lulc)

    # calculate the suitable area in sqkm per grid_cell (fi * ai)
    suitable_area_csp_sqkm = suitability_factor_csp * grid_cell_area[::-1, :]

    out_suit_f_csp = os.path.join(output_directory, 'solar_csp_suitability_factor.npy')
    np.save(out_suit_f_csp, suitability_factor_csp)

    out_suit_area_csp = os.path.join(output_directory, 'solar_csp_suitable_area_sqkm.npy')
    np.save(out_suit_area_csp, suitable_area_csp_sqkm)

    return suitable_area_csp_sqkm

# def calc_technical_potential_solar_pv(temp_ambient_k: np.ndarray,
#                                       radiation: np.ndarray,
#                                       wind_speed: np.ndarray,
#                                       standard_panel_eff: float = 0.17,
#                                       temp_ref_c: float = 25.0,
#                                       eff_response_coef: float = -0.005,
#                                       thermal_coef1: float = 4.3,
#                                       thermal_coef2: float = 0.943,
#                                       thermal_coef3: float = 0.028,
#                                       thermal_coef4: float = -1.528) -> np.ndarray:


def calc_technical_potential_solar_pv(r_rsds,
                                      r_wind,
                                      r_tas,
                                      suit_sqkm_raster,
                                      yr_hours,
                                      output_directory,
                                      target_year):

    """
    Calculates the solar PV technical potential as an array in kWh per year.

    Source: Gernaat et al. 2021; https://doi.org/10.1038/s41558-020-00949-9
    """

    # Important variables
    # Central assumptions (based on Gernaat et al. 2021)
    N_lpv = 0.47  # land-use factor or packing factor
    PR = 0.85  # Performance ratio
    # Sensitivities
    # N_lpv = 1.0               # land-use factor or packing factor   - Hoogwick 2004
    # N_lpv = 0.20              # land-use factor or packing factor   - Deng et al. 2015
    # N_lpv = 0.30              # land-use factor or packing factor   - Deng et al. 2015
    # PR    = 0.75              # Performance ratio                   - Deng et al. 2015
    # PR    = 0.80              # Performance ratio                   - Dupont et al. 2020
    # PR    = 0.90              # Performance ratio                   - High case

    # camputes the daily N_pv array
    N_pv_daily = adjust_pv_panel_eff_for_atm_condition(r_tas, r_rsds, r_wind)

    # camputes the yearly mean N_pv array
    N_pv = np.mean(N_pv_daily, axis=0)

    # save N_pv for debug
    raster = os.path.join(output_directory, 'N_pv_' + str(target_year) + '.npy')
    np.save(raster, N_pv)

    # camputes the yearly mean solar radiation array
    solar_rad = np.mean(r_rsds, axis=0)

    # computes the solar PV technical potential
    solar_potential_KWh = 1000.0 * solar_rad * suit_sqkm_raster * yr_hours * N_lpv * N_pv * PR

    # save the solar PV technical potential files
    solar_potential_raster = os.path.join(output_directory, 'solar_PV_technical_potential_' + str(target_year) + '.npy')
    np.save(solar_potential_raster, solar_potential_KWh)

    # OPTIONAL - define a general PV power production term
    # -> (based on Crook et al. 2011 - https://doi.org/10.1039/C1EE01495A)
    # Not used to compute the technical potential but it can be used for
    # -> climate impact studies as in Crook et al. 2011
    # PV_prod = solar_rad * N_pv     # W/m^2
    # PV_prod_raster = os.path.join(output_directory, 'PV_prod_'+str(target_year)+'.npy')
    # np.save(PV_prod_raster, PV_prod)

    return solar_potential_KWh


def calc_technical_potential_solar_CSP(r_rsds, r_tas, suit_sqkm_raster, yr_hours, output_directory, target_year):
    """Calculates the solar CSP technical potential as an array in kWh per year.
       Reference: Gernaat et al. 2021; https://doi.org/10.1038/s41558-020-00949-9 """

    # Important variables
    N_lcsp = 0.37  # CSP land-use factor or packing factor - Central
    # N_lcsp = 0.135              # Sensitivity based on Dupont et al. 2020
    # N_lcsp = 0.20               # Sensitivity based on Deng et al. 2015
    # N_lcsp = 0.50               # Sensitivity (high case)

    # computes the yearly mean solar radiation array (W/m^2)
    solar_rad = np.mean(r_rsds, axis=0)

    # convert solar_rad from W/m2 to KWh/m2
    solar_rad_KWh = (solar_rad * yr_hours) / 1000.0

    solar_rad_raster = os.path.join(output_directory, 'solar_rad_KWh_m2_' + str(target_year) + '.npy')
    np.save(solar_rad_raster, solar_rad_KWh)

    # computes FLH
    FLH = compute_full_load_hours_for_csp(solar_rad_KWh)

    # print FLH for debug
    FLH_raster = os.path.join(output_directory, 'FLH_' + str(target_year) + '.npy')
    np.save(FLH_raster, FLH)

    # computes the daily CSP_eff
    N_csp_daily = compute_csp_eff(r_tas, r_rsds)

    # save N_csp_daily for debug
    # raster = os.path.join(output_directory, 'N_csp_daily_'+str(target_year)+'.npy')
    # np.save(raster, N_csp_daily)

    # computes the yearly mean N_csp array
    N_csp = np.mean(N_csp_daily, axis=0)

    # save N_csp for debug
    # raster = os.path.join(output_directory, 'N_csp_'+str(target_year)+'.npy')
    # np.save(raster, N_csp)

    # computes the solar csp technical potential
    suit_sqm_raster = suit_sqkm_raster * 1.0e6  # change from km^2 to m^2
    solar_potential_KWh = np.zeros_like(solar_rad)
    # Avoid division by zero
    idx = np.where(FLH > 0.0)
    solar_potential_KWh[idx] = solar_rad_KWh[idx] * suit_sqm_raster[idx] * yr_hours * N_lcsp * (N_csp[idx] / FLH[idx])

    # save the solar CSP technical potential files
    solar_potential_raster = os.path.join(output_directory,
                                          'solar_csp_technical_potential_' + str(target_year) + '.npy')
    np.save(solar_potential_raster, solar_potential_KWh)

    # OPTIONAL - define a general CSP power production term
    # -> (based on Crook et al. 2011 - https://doi.org/10.1039/C1EE01495A)
    # Not used to compute the technical potential but it can be used for
    # -> climate impact studies as in Crook et al. 2011
    # CSP_prod = solar_rad * N_csp     # W/m^2
    # CSP_prod_raster = os.path.join(output_directory, 'CSP_prod_'+str(target_year)+'.npy')
    # np.save(CSP_prod_raster, CSP_prod)

    return solar_potential_KWh


def calc_technical_potential(r_wind, r_ps, r_tas, r_huss, suit_sqkm_raster, power_density, yr_hours,
                             n_avail, n_array, p_rated, output_directory, target_year, wind_turbname):
    """Calculates the wind technical potential as an array in kWh per year.
       References: Eurek et al. 2017, http://dx.doi.org/10.1016/j.eneco.2016.11.015
                   Karnauskas et al. 2018, https://doi.org/10.1038/s41561-017-0029-9
                   Rinne et al. 2018, https://doi.org/10.1038/s41560-018-0137-9 """

    # calculate wind speed at the hub height
    wind = r_wind
    wind_speed = extrap_wind_speed_at_hub_height(wind, 125)  # Central based on Rinne et al. 2018
    # wind_speed = extrap_wind_speed_at_hub_height(wind, 100) # Sensitivity based on Rinne et al. 2018
    # wind_speed = extrap_wind_speed_at_hub_height(wind, 75)  # Sensitivity based on Rinne et al. 2018
    # wind_speed = extrap_wind_speed_at_hub_height(wind, 150) # Sensitivity based on Rinne et al. 2018

    # rho - dry air density (kg/m3)
    ps = r_ps
    tas = r_tas
    rho_d = dry_air_density_ideal(ps, tas)

    # rho_m - air density corrected for humidity (kg/m3)
    # Note - this correction can be omitted if specific humidity data is unavailable since this is
    # a minor correction.
    huss = r_huss
    rho_m = dry_air_density_humidity(rho_d, huss)

    # wadj - wind speed adjusted for air density (m/s)
    # Note - if the correction for humidity is not made, rho_m is replaced by rho_d (dry air density)
    wadj = adjust_wind_speed_for_air_density(wind_speed, rho_m)

    # Compute wind power at the hub height H using power curve from the representative turbine
    p_daily = compute_wind_power(wadj, wind_turbname)
    # sensitivities - other turbine models
    # p_daily = power_curve_vestas_v90_2000(wadj)
    # p_daily = power_curve_GE_2500(wadj)  # KW
    # p_daily = power_curve_E101_3050(wadj)

    # daily wind power to yearly mean wind power
    p = np.mean(p_daily, axis=0)

    # write to .npy
    p_raster = os.path.join(output_directory, 'wind_power_kw_' + str(target_year) + '.npy')
    np.save(p_raster, p)

    # read in suitable sqkm per gridcell raster
    arr_suit_area = suit_sqkm_raster

    # compute capacity factor (CF)
    CF = ((p / 1000.0) * n_avail * n_array) / p_rated

    # calculate technical potential per gridcell
    land_potential_MWh = arr_suit_area * power_density * yr_hours * CF

    # MWh to KWh
    land_potential = land_potential_MWh * 1000.0

    land_potential_raster = os.path.join(output_directory, 'wind_technical_potential_' + str(target_year) + '.npy')
    np.save(land_potential_raster, land_potential)

    CF_raster = os.path.join(output_directory, 'CF_' + str(target_year) + '.npy')
    np.save(CF_raster, CF)

    return land_potential
