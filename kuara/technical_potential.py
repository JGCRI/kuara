import os
import calendar
import pkg_resources

import numpy as np
import pandas as pd
from scipy import interpolate
import xarray as xr
import rasterio


def power_law(wind_speed, hub_height_m=100):
    """Extrapolates wind speed at 10m to the turbine hub height (H) by using the power law.

    Sources: Karnauskas et al. 2018, Nature Geoscience, https://doi.org/10.1038/s41561-017-0029-9

    :param wind_speed:                              w - wind speed at 10m (m/s)

    :param hub_height_m:                            H - Turbine hub height (m)

    :return:                                        W_h - wind speed at the turbine hub height H (m/s)

    """

    return wind_speed * ((hub_height_m / 10.0) ** (1.0 / 7.0))


def dry_air_density_ideal(pressure, temperature):
    """Computes dry air density based on the ideal gas law.

    Source: Karnauskas et al. 2018, Nature Geoscience, https://doi.org/10.1038/s41561-017-0029-9

        inputs:   p   - pressure (Pascal = J/m3)
                  t   - temperature (K)

        output:   rho - dry air density (kg/m3)

    """

    rd = 287.058  # J / Kg * K

    return pressure / (rd * temperature)


def dry_air_density_humidity(rho_d, q):
    """Corrects dry air density for humidity
        Source: Karnauskas et al. 2018, Nature Geoscience, https://doi.org/10.1038/s41561-017-0029-9

        inputs:   rho_d  - dry air density (kg/m3)
                  q      - surface specific humidity (Kg/Kg)

        output:   rho_m - air density corrected for humidity (kg/m3)

    """

    rho_m = rho_d * ((1.0 + q) / (1.0 + 1.609 * q))
    return rho_m


def wind_speed_adjusted(w, dens):
    """Designed to account for the differences in air density at the rotor elevations as compared with
        standard (sea level) conditions.
        Sources: Karnauskas et al. 2018, Nature Geoscience, https://doi.org/10.1038/s41561-017-0029-9;
                 IEC61400-12-1 (2005)

        inputs:   w    - wind speed (m/s)
                  dens - air density (kg/m3)

        outputs:  wadj    - wind speed adjusted for air density (m/s)

    """

    rho_std = 1.225
    wadj = w * ((dens / rho_std) ** (1.0 / 3.0))
    return wadj


def compute_wind_power(wind_speed_arr: np.ndarray,
                       wind_to_fit: np.ndarray,
                       power_to_fit: np.ndarray,
                       min_watt_hr: float,
                       max_watt_hr: float) -> np.ndarray:
    """Compute wind power for the target turbine model.

    :param wind_speed_arr:               array of wind speeds at the turbine hub height H (m/s)
    :type wind_speed_arr:                numpy array

    :param wind_to_fit:                 wind data points to fit power curve
    :type wind_to_fit:                  numpy array

    :param power_to_fit:                 power data points to fit power curve
    :type power_to_fit:                  numpy array

    :param min_watt_hr:                 Minimum power in watt hours.
    :type min_watt_hr:                  float

    :param max_watt_hr:                 Maximum power in watt hours.
    :type max_watt_hr:                  float

    """

    # Creating np.array for power filled with zeros (p is the output array)
    # power_arr = np.zeros_like(wind_speed_arr)

    # filtering wind input data for the range btw max and min speed for a turbine type
    idx_wind_filt = np.where((wind_speed_arr <= max_watt_hr) * (wind_speed_arr >= min_watt_hr))
    wind_filt = wind_speed_arr[idx_wind_filt]

    # linear interpolation
    f = interpolate.interp1d(wind_to_fit, power_to_fit, fill_value="extrapolate")  # (x, y) pair

    # calculating wind power as a linear interpolation between points in the power curve
    power_interp = f(wind_filt)

    # Filling out the output array p for wind in the btw 3.0 - 15.0 m/s range
    # power_arr = power_interp

    return power_interp


def wind_power_curve(W_h, wind_turbname):
    """Compute wind power for any turbine.

        inputs:   W_h - array of wind speeds at the turbine hub height H (m/s)

        outputs:  power_arr   - array of wind power at the turbine hub height H (kW)

    """

    # import wind power data
    global power_arr
    wind_power_file = pkg_resources.resource_filename('kuara', 'data/wind_power_data_points_to_fit.csv')
    df_wind_power_to_fit = pd.read_csv(wind_power_file, header=0)

    # Wind points of the wind power curve
    wind_to_fit = np.array(df_wind_power_to_fit['wind_'+wind_turbname])
    
    # Wind power points of the wind power curve
    power_to_fit = np.array(df_wind_power_to_fit['power_'+wind_turbname])
    
    # Exclude nan if any
    wind_to_fit = wind_to_fit[~np.isnan(wind_to_fit)]
    power_to_fit = power_to_fit[~np.isnan(power_to_fit)]

    if wind_turbname == 'vestas_v136_3450':
        """Computes wind power assuming characteristics of the wind turbine model V136-3.45 MW
           (indicated in low- and medium-wind conditions: 
           https://www.vestas.com/en/products/4-mw-platform/v136-_3_45_mw#!about)
            Specifications and power curve available at:
            https://www.thewindpower.net/turbine_en_1074_vestas_v136-3450.php
            brochure: 
            http://nozebra.ipapercms.dk/Vestas/Communication/Productbrochure/4MWbrochure/4MWProductBrochure/?page=14
            Commissioning: 2015
        """

        power_arr = compute_wind_power(wind_speed_arr=W_h,
                                       wind_to_fit=wind_to_fit,
                                       power_to_fit=power_to_fit,
                                       min_watt_hr=2.5,
                                       max_watt_hr=22.0)

    elif wind_turbname == 'vestas_v90_2000':
        """Computes wind power assuming characteristics of the wind turbine model vestas_v90_2000 
           (indicated in low- and medium-wind conditions: 
           https://www.vestas.com/en/products/4-mw-platform/v136-_3_45_mw#!about)
            Specifications and power curve available at:
            https://www.thewindpower.net/turbine_en_32_vestas_v90-2000.php
            Commissioning: 2004

        """ 
        power_arr = compute_wind_power(wind_speed_arr=W_h,
                                       wind_to_fit=wind_to_fit,
                                       power_to_fit=power_to_fit,
                                       min_watt_hr=3.0,
                                       max_watt_hr=25.0)

    elif wind_turbname == 'GE_2500':
        """Computes wind power assuming characteristics of the wind turbine model GE 2.5-100 MW
           (indicated in low- and medium-wind conditions: https://www.ge.com/in/wind-energy/2.5-MW-wind-turbine
            Specifications and power curve available at:
            https://www.thewindpower.net/turbine_en_382_ge-energy_2.5-100.php
            Commissioning: 2006
        """

        power_arr = compute_wind_power(wind_speed_arr=W_h,
                                       wind_to_fit=wind_to_fit,
                                       power_to_fit=power_to_fit,
                                       min_watt_hr=3.0,
                                       max_watt_hr=25.0)

    elif wind_turbname == 'E101_3050':
        """Computes wind power assuming characteristics of the wind turbine model E-101 3.05 MW
           Indicated for medium-wind conditions
           Specifications and power curve available at:
           https://www.thewindpower.net/turbine_en_924_enercon_e101-3050.php and
           https://www.enercon.de/fileadmin/Redakteur/Medien-Portal/broschueren/pdf/en/ENERCON_Produkt_en_06_2015.pdf
           Commissioning: 2012
        """

        power_arr = compute_wind_power(wind_speed_arr=W_h,
                                       wind_to_fit=wind_to_fit,
                                       power_to_fit=power_to_fit,
                                       min_watt_hr=2.0,
                                       max_watt_hr=25.0)

    elif wind_turbname == 'Gamesa_G114_2000':
        """Computes wind power assuming characteristics of the wind turbine model Gamesa G114-2000
            indicated for class III (low winds)
            https://en.wind-turbine-models.com/turbines/428-gamesa-g114-2.0mw
            https://www.thewindpower.net/turbine_en_860_gamesa_g114-2000.php

        """

        power_arr = compute_wind_power(wind_speed_arr=W_h,
                                       wind_to_fit=wind_to_fit,
                                       power_to_fit=power_to_fit,
                                       min_watt_hr=2.5,
                                       max_watt_hr=25.0)

    elif wind_turbname == 'IEC_classII_3500':
        """Computes wind power assuming characteristics of the: Wind Turbine 3.5 MW (IEC) class II composite
            Reference: Eurek et al. 2017,  http://dx.doi.org/10.1016/j.eneco.2016.11.015

        """

        power_arr = compute_wind_power(wind_speed_arr=W_h,
                                       wind_to_fit=wind_to_fit,
                                       power_to_fit=power_to_fit,
                                       min_watt_hr=4.0,
                                       max_watt_hr=15.0)

        # Filling out the out array power_arr for wind wind in the 15.0 - 25.0 m/s range
        power_arr = np.where((W_h <= 25.0) * (W_h >= 15.0), 3500.0, power_arr)

    elif wind_turbname == 'GE1500':
        """Computes wind power assuming characteristics of the: Wind Turbine Model GE 1.5s
            Specifications and power curve available at:
            https://www.en.wind-turbine-models.com/turbines/565-general-electric-ge-1.5s

        """

        # Generate Sixth Order Fit Coef
        coef_lin = np.polyfit(wind_to_fit, power_to_fit, 6)

        # Generate Sixth Order Fit Formula
        lfit = np.poly1d(coef_lin)

        # Aplying to the wind dataset
        res = np.ma.where((W_h < 13.5) * (W_h > 3.5))
        power_arr = lfit(W_h[res])

        # res = np.ma.where((W_h <= 25.0) * (W_h >= 13.5))
        # power_arr = 1500.0  # rated power
        power_arr = np.where((W_h <= 25.0) * (W_h >= 13.5), 1500.0, power_arr)

    return power_arr


def compute_solar_to_electric_eff(Temp_K, Rad, Sfcwind):
    """
    Computes the Solar-to-Electric efficiency, which accounts for the PV cell efficiency of the conversion
    of solar radiation into electricity under the operating climate conditions of the plant site.
    Source: Gernaat et al. 2021; https://doi.org/10.1038/s41558-020-00949-9
    inputs: Temp_K    - array of surface air temperature (K)
            Rad     - array of surface-downwelling shortwave radiation (W/m^2)
            Sfcwind - array of wind speeds at 10m height (m/s)
    output: Solar-to-Electric efficiency (N_pv) - array of the same shape as the original input data (adimensional)
    """
    # Initializing key parameters
    N_panel = 0.17  # i.e., 17%
    T_stc = 25.0  # deg C
    thermal_coef = -0.005  # (deg C)^-1
    c1 = 4.3  # deg C
    c2 = 0.943  # adimensional
    c3 = 0.028  # deg C * m^2 * W^-1
    c4 = -1.528  # deg C * s * m^-1
    # Converting Temp from K to deg C
    Temp_C = Temp_K - 273.15
    # Computing the PV cell temperature
    Tcell = c1 + c2 * Temp_C + c3 * Rad + c4 * Sfcwind
    # Computing Solar-to-Electric efficiency (N_pv)
    N_pv = N_panel * (1.0 + thermal_coef * (Tcell - T_stc))
    #
    return N_pv


def compute_FLH(rad):
    """
    Checks the minimum feasibility threshold level for CSP operation regarding solar
    radiation and computes the FLH (full load hours) as a function of the incident solar
    radiation for a CSP plant SM 2.  Minimum taken as 3000 Wh/m2/day (or an average of
    300 W/m2 over a 10-hour solar day). This translates into 1095 kWh/m2/year.
    Sources: Koberle et al. 2015. http://dx.doi.org/10.1016/j.energy.2015.05.145
             Gernaat et al. 2021; https://doi.org/10.1038/s41558-020-00949-9
    input:   rad         - a numpy array with annual average surface solar radiation (KWh/m2/year)
    outputs: FLH         - a numpy array (same dimemsion of the input array) with Full Load Hours (hours)
    """
    FLH = np.zeros_like(rad)
    FLH = np.where(rad > 2800.0, 5260.0, FLH)
    idx_FLH = np.where((rad >= 1095.0) * (rad <= 2800.0))
    FLH[idx_FLH] = 1.83 * rad[idx_FLH] + 150.0
    FLH = np.where(FLH > 5260.0, 5260.0, FLH)

    return FLH


def compute_CSP_eff(Temp_K, Rad):
    """
    Computes the thermal efficiency of a CSP system.
    Source: Gernaat et al. 2021; https://doi.org/10.1038/s41558-020-00949-9
    inputs: Temp_K   - array of surface air temperature (K)
            Rad     - array of surface-downwelling shortwave radiation (W/m^2)
    output: CSP efficiency (N_csp) - array of the same shape as the original input data (adimensional)
    """
    # Initializing key parameters
    N_rank = 0.40  # i.e., 40%    # central assumptions
    # N_rank  = 0.37     # i.e., 37%    # Huixing et al 2020
    # N_rank  = 0.42     # i.e., 42%    # Huixing et al 2020
    k0 = 0.762  # adimensional
    k1 = 0.2125  # W m^−2 deg C^−1
    T_fluid = 115.0  # deg C
    # Converting Temp from K to deg C
    Temp_C = Temp_K - 273.15

    # Initialize the array N_csp with 0s
    N_csp = np.zeros_like(Rad)

    # Avoid division by zero (ocean cells and a group of land cells at higher latitudes in the Winter
    # have 0.0 values for the Rad array). Setting a cutoff value of 1.0 W/m2 does not affect the
    # estimate of solar CSP potential since this value is much lower than the minimum operational
    # requirement of 300 W/m2.
    idx = np.where(Rad >= 1.0)

    # Use indexes above (idx) to compute the CSP efficiency (N_csp)
    N_csp[idx] = N_rank * (k0 - (k1 * (T_fluid - Temp_C[idx])) / Rad[idx])

    # Filtering negative values that can occur at higher latitudes in the Winter
    N_csp = np.where(N_csp <= 0.0, 0.0, N_csp)

    return N_csp


def read_climate_data(nc_file, variable, target_year, groupby_freq='Y', time_dim='time'):
    """Read in NetCDF file and get the mean of the desired time period as an array.

    :param nc_file:  Full path with file name and extension to the input NetCDF file
    :type nc_file: str

    :param variable:  Name of the target variable in the NetCDF file
    :type variable: str

    :param target_year: Target year to process in YYYY format.
    :type target_year:  int; str

    :param groupby_freq:  The datetime part to group by for output statistics.
                        Default:  'Y' for year.
    :type groupby_freq:  str

    :param time_dim: Time dimension name in the NetCDF file.
    :type time_dim: str

    :return:   A n-dimensional array where values for [time, lat, lon].  If only one time then a
                2D array of values for [lat, lon] will be returned.


    """
    yr = str(target_year)

    # read in NetCDF file
    ds = xr.open_dataset(nc_file)

    # get desired time slice from data
    dsx = ds.sel(time=slice(yr, yr))

    # extract daily data for target_year
    arr = dsx.variables[variable][:, :, :].values

    return arr


def process_climate_solar(nc_rsds, target_year, output_directory, rsds_var='rsdsAdjust'):
    """Process each required climate NetCDF file."""

    arr_rsds = read_climate_data(nc_rsds, rsds_var, target_year)
    # Handling NaN values: NaN values correspond to ocean cells with no values.
    # This treatment does not affect results of land wind potential but avoids
    # "RuntimeWarning" when arithmetic operations are attempted with NaN values.
    idx_nan = np.isnan(arr_rsds)
    arr_rsds[idx_nan] = 0.0

    # saving climate variables to .npy
    # daily to yearly mean
    arr_rsds_mean = np.mean(arr_rsds, axis=0)
    arr_rsds_raster = os.path.join(output_directory, 'solar_radiation_W_m2_' + str(target_year) + '.npy')
    np.save(arr_rsds_raster, arr_rsds_mean)

    return arr_rsds


def process_climate(nc_wind, nc_tas, nc_ps, nc_huss, target_year, output_directory,
                    wind_var='sfcWind', tas_var='tas', ps_var='ps',
                    huss_var='huss'):
    """Process each required climate NetCDF file."""

    # print('### process wind ###')
    arr_wind = read_climate_data(nc_wind, wind_var, target_year)
    # Handling NaN values: NaN values correspond to ocean cells with no values.
    # This treatment does not affect results of land wind potential but avoids
    # "RuntimeWarning" when arithmetic operations are attempted with NaN values.
    idx_nan = np.isnan(arr_wind)
    arr_wind[idx_nan] = 0.0

    # print('### process tas ###')
    arr_tas = read_climate_data(nc_tas, tas_var, target_year)
    arr_tas[idx_nan] = -999.0  # cannot be 0.0, otherwise "RuntimeWarning" is raised

    # print('### process ps ###')
    arr_ps = read_climate_data(nc_ps, ps_var, target_year)
    arr_ps[idx_nan] = 0.0

    # print('### process huss ###')
    arr_huss = read_climate_data(nc_huss, huss_var, target_year)
    arr_huss[idx_nan] = 0.0

    # saving climate variables to .npy
    # daily to yearly mean
    arr_wind_mean = np.mean(arr_wind, axis=0)
    arr_wind_raster = os.path.join(output_directory, 'wind_speed_ms_' + str(target_year) + '.npy')
    np.save(arr_wind_raster, arr_wind_mean)

    # daily to yearly mean
    arr_tas_mean = np.mean(arr_tas, axis=0)
    arr_tas_raster = os.path.join(output_directory, 'tas_degK_' + str(target_year) + '.npy')
    np.save(arr_tas_raster, arr_tas_mean)

    # daily to yearly mean
    # arr_ps_mean = np.mean(arr_ps, axis=0)
    # arr_ps_raster = os.path.join(output_directory, 'ps_Pa_'+str(target_year)+'.npy')
    # np.save(arr_ps_raster, arr_ps_mean)

    # daily to yearly mean
    # arr_huss_mean = np.mean(arr_huss, axis=0)
    # arr_huss_raster = os.path.join(output_directory, 'huss_kgperkg_'+str(target_year)+'.npy')
    # np.save(arr_huss_raster, arr_huss_mean)

    return arr_wind, arr_tas, arr_ps, arr_huss


def process_elevation(r_file):
    r = rasterio.open(r_file)

    return np.where(r.read(1) > 2500, 0, 1)


def process_slope(r_file):
    r = rasterio.open(r_file)

    return np.where(r.read(1) > 20, 0, 1)


def process_protected(r_file):
    r = rasterio.open(r_file)

    return np.where(r.read(1) < 0.0, 1, 0)


def process_permafrost(r_file):
    r = rasterio.open(r_file)

    return np.where(r.read(1) >= 0.1, 0, 1)


def process_lulc(r_file):
    # central assumptions - based on Eurek et al. 2017

    r = rasterio.open(r_file)

    arr = r.read(1).astype(np.float64)

    arr = np.where(arr == 11, 0, arr)
    arr = np.where(arr == 14, 0.7, arr)
    arr = np.where(arr == 20, 0.7, arr)
    arr = np.where(arr == 30, 0.7, arr)
    arr = np.where(arr == 40, 0.1, arr)
    arr = np.where(arr == 50, 0.1, arr)
    arr = np.where(arr == 60, 0.1, arr)
    arr = np.where(arr == 70, 0.1, arr)
    arr = np.where(arr == 90, 0.1, arr)
    arr = np.where(arr == 100, 0.1, arr)
    arr = np.where(arr == 110, 0.5, arr)
    arr = np.where(arr == 120, 0.65, arr)
    arr = np.where(arr == 130, 0.5, arr)
    arr = np.where(arr == 140, 0.8, arr)
    arr = np.where(arr == 150, 0.9, arr)
    arr = np.where(arr == 160, 0, arr)
    arr = np.where(arr == 170, 0, arr)
    arr = np.where(arr == 180, 0, arr)
    arr = np.where(arr == 190, 0, arr)
    arr = np.where(arr == 200, 0.9, arr)
    arr = np.where(arr == 210, 0, arr)
    arr = np.where(arr == 220, 0, arr)
    arr = np.where(arr == 230, 0, arr)

    return arr


'''
def process_lulc(r_file):
    # wind high suitability (S_high) - based on Lu et al. 2009

    r = rasterio.open(r_file)

    arr = r.read(1).astype(np.float64)

    arr = np.where(arr == 11, 0, arr)
    arr = np.where(arr == 14, 1.0, arr)
    arr = np.where(arr == 20, 1.0, arr)
    arr = np.where(arr == 30, 1.0, arr)
    arr = np.where(arr == 40, 0, arr)
    arr = np.where(arr == 50, 0, arr)
    arr = np.where(arr == 60, 0, arr)
    arr = np.where(arr == 70, 0, arr)
    arr = np.where(arr == 90, 0, arr)
    arr = np.where(arr == 100, 0, arr)
    arr = np.where(arr == 110, 1.0, arr)
    arr = np.where(arr == 120, 1.0, arr)
    arr = np.where(arr == 130, 1.0, arr)
    arr = np.where(arr == 140, 1.0, arr)
    arr = np.where(arr == 150, 1.0, arr)
    arr = np.where(arr == 160, 0, arr)
    arr = np.where(arr == 170, 0, arr)
    arr = np.where(arr == 180, 0, arr)
    arr = np.where(arr == 190, 0, arr)
    arr = np.where(arr == 200, 1.0, arr)
    arr = np.where(arr == 210, 0, arr)
    arr = np.where(arr == 220, 0, arr)
    arr = np.where(arr == 230, 0, arr)

    return arr


def process_lulc(r_file):
    # wind low suitability 1 (S_low) - based on Zhou et al. 2012 (low case)

    r = rasterio.open(r_file)

    arr = r.read(1).astype(np.float64)

    arr = np.where(arr == 11, 0, arr)
    arr = np.where(arr == 14, 0.6, arr)
    arr = np.where(arr == 20, 0.6, arr)
    arr = np.where(arr == 30, 0.6, arr)
    arr = np.where(arr == 40, 0.0, arr)
    arr = np.where(arr == 50, 0.0, arr)
    arr = np.where(arr == 60, 0.0, arr)
    arr = np.where(arr == 70, 0.0, arr)
    arr = np.where(arr == 90, 0.0, arr)
    arr = np.where(arr == 100, 0.0, arr)
    arr = np.where(arr == 110, 0.1, arr)
    arr = np.where(arr == 120, 0.1, arr)
    arr = np.where(arr == 130, 0.2, arr)
    arr = np.where(arr == 140, 0.2, arr)
    arr = np.where(arr == 150, 0.2, arr)
    arr = np.where(arr == 160, 0, arr)
    arr = np.where(arr == 170, 0, arr)
    arr = np.where(arr == 180, 0, arr)
    arr = np.where(arr == 190, 0, arr)
    arr = np.where(arr == 200, 0.1, arr)
    arr = np.where(arr == 210, 0, arr)
    arr = np.where(arr == 220, 0, arr)
    arr = np.where(arr == 230, 0, arr)

    return arr


def process_lulc(r_file):
    # wind low suitability 2 (S_low_II) - based on Deng et al. 2015 (low case)

    r = rasterio.open(r_file)

    arr = r.read(1).astype(np.float64)

    arr = np.where(arr == 11, 0, arr)
    arr = np.where(arr == 14, 0.03, arr)
    arr = np.where(arr == 20, 0.03, arr)
    arr = np.where(arr == 30, 0.03, arr)
    arr = np.where(arr == 40, 0.005, arr)
    arr = np.where(arr == 50, 0.005, arr)
    arr = np.where(arr == 60, 0.005, arr)
    arr = np.where(arr == 70, 0.005, arr)
    arr = np.where(arr == 90, 0.005, arr)
    arr = np.where(arr == 100, 0.005, arr)
    arr = np.where(arr == 110, 0.03, arr)
    arr = np.where(arr == 120, 0.03, arr)
    arr = np.where(arr == 130, 0.03, arr)
    arr = np.where(arr == 140, 0.03, arr)
    arr = np.where(arr == 150, 0.03, arr)
    arr = np.where(arr == 160, 0, arr)
    arr = np.where(arr == 170, 0, arr)
    arr = np.where(arr == 180, 0, arr)
    arr = np.where(arr == 190, 0, arr)
    arr = np.where(arr == 200, 0.03, arr)
    arr = np.where(arr == 210, 0, arr)
    arr = np.where(arr == 220, 0, arr)
    arr = np.where(arr == 230, 0, arr)

    return arr
'''


def process_elevation_solar(r_file):
    # Following Deng et al. 2015 (no constraint for altitude for solar)

    r = rasterio.open(r_file)

    return np.where(r.read(1) > 2500, 1, 1)


def process_slope_solar_PV(r_file):
    # Following Deng et al. 2015

    r = rasterio.open(r_file)

    return np.where(r.read(1) > 27, 0, 1)


def process_slope_solar_CSP(r_file):
    # Following Deng et al. 2015

    r = rasterio.open(r_file)

    return np.where(r.read(1) > 4, 0, 1)


def process_permafrost_solar(r_file):
    r = rasterio.open(r_file)

    return np.where(r.read(1) >= 0.1, 1, 1)


def process_lulc_solar(r_file):
    # Central assumptions - Suitability factors based on Gernaat et al. 2021 and Korfiati et al. 2016

    r = rasterio.open(r_file)

    arr = r.read(1).astype(np.float64)

    arr = np.where(arr == 11, 0.0, arr)
    arr = np.where(arr == 14, 0.01, arr)
    arr = np.where(arr == 20, 0.01, arr)
    arr = np.where(arr == 30, 0.01, arr)
    arr = np.where(arr == 40, 0.0, arr)
    arr = np.where(arr == 50, 0.0, arr)
    arr = np.where(arr == 60, 0.0, arr)
    arr = np.where(arr == 70, 0.0, arr)
    arr = np.where(arr == 90, 0.0, arr)
    arr = np.where(arr == 100, 0.0, arr)
    arr = np.where(arr == 110, 0.01, arr)
    arr = np.where(arr == 120, 0.01, arr)
    arr = np.where(arr == 130, 0.01, arr)
    arr = np.where(arr == 140, 0.01, arr)
    arr = np.where(arr == 150, 0.01, arr)
    arr = np.where(arr == 160, 0.0, arr)
    arr = np.where(arr == 170, 0.0, arr)
    arr = np.where(arr == 180, 0.0, arr)
    arr = np.where(arr == 190, 0.0, arr)
    arr = np.where(arr == 200, 0.05, arr)
    arr = np.where(arr == 210, 0.0, arr)
    arr = np.where(arr == 220, 0.0, arr)
    arr = np.where(arr == 230, 0.0, arr)

    return arr


'''

def process_lulc_solar(r_file):
    # Solar low suitability 2 (S_low_II) - based on Deng et al. 2015 (medium case)

    r = rasterio.open(r_file)

    arr = r.read(1).astype(np.float64)

    arr = np.where(arr == 11, 0.0, arr)
    arr = np.where(arr == 14, 0.005, arr)
    arr = np.where(arr == 20, 0.005, arr)
    arr = np.where(arr == 30, 0.005, arr)
    arr = np.where(arr == 40, 0.0, arr)
    arr = np.where(arr == 50, 0.0, arr)
    arr = np.where(arr == 60, 0.0, arr)
    arr = np.where(arr == 70, 0.0, arr)
    arr = np.where(arr == 90, 0.0, arr)
    arr = np.where(arr == 100, 0.0, arr)
    arr = np.where(arr == 110, 0.01, arr)
    arr = np.where(arr == 120, 0.01, arr)
    arr = np.where(arr == 130, 0.01, arr)
    arr = np.where(arr == 140, 0.01, arr)
    arr = np.where(arr == 150, 0.01, arr)
    arr = np.where(arr == 160, 0.0, arr)
    arr = np.where(arr == 170, 0.0, arr)
    arr = np.where(arr == 180, 0.0, arr)
    arr = np.where(arr == 190, 0.0, arr)
    arr = np.where(arr == 200, 0.01, arr)
    arr = np.where(arr == 210, 0.0, arr)
    arr = np.where(arr == 220, 0.0, arr)
    arr = np.where(arr == 230, 0.0, arr)

    return arr


def process_lulc_solar(r_file):
    # Solar low suitability (S_low) - based on Deng et al. 2015 (low case)

    r = rasterio.open(r_file)

    arr = r.read(1).astype(np.float64)

    arr = np.where(arr == 11, 0.0, arr)
    arr = np.where(arr == 14, 0.001, arr)
    arr = np.where(arr == 20, 0.001, arr)
    arr = np.where(arr == 30, 0.001, arr)
    arr = np.where(arr == 40, 0.0, arr)
    arr = np.where(arr == 50, 0.0, arr)
    arr = np.where(arr == 60, 0.0, arr)
    arr = np.where(arr == 70, 0.0, arr)
    arr = np.where(arr == 90, 0.0, arr)
    arr = np.where(arr == 100, 0.0, arr)
    arr = np.where(arr == 110, 0.005, arr)
    arr = np.where(arr == 120, 0.005, arr)
    arr = np.where(arr == 130, 0.005, arr)
    arr = np.where(arr == 140, 0.005, arr)
    arr = np.where(arr == 150, 0.005, arr)
    arr = np.where(arr == 160, 0.0, arr)
    arr = np.where(arr == 170, 0.0, arr)
    arr = np.where(arr == 180, 0.0, arr)
    arr = np.where(arr == 190, 0.0, arr)
    arr = np.where(arr == 200, 0.005, arr)
    arr = np.where(arr == 210, 0.0, arr)
    arr = np.where(arr == 220, 0.0, arr)
    arr = np.where(arr == 230, 0.0, arr)

    return arr


def process_lulc_solar(r_file):
    # Solar high suitability (S_high) - based on Dupont et al. 2020 (high case)

    r = rasterio.open(r_file)

    arr = r.read(1).astype(np.float64)

    arr = np.where(arr == 11, 0.0, arr)
    arr = np.where(arr == 14, 0.1, arr)
    arr = np.where(arr == 20, 0.05, arr)
    arr = np.where(arr == 30, 0.05, arr)
    arr = np.where(arr == 40, 0.0, arr)
    arr = np.where(arr == 50, 0.0, arr)
    arr = np.where(arr == 60, 0.0, arr)
    arr = np.where(arr == 70, 0.0, arr)
    arr = np.where(arr == 90, 0.0, arr)
    arr = np.where(arr == 100, 0.0, arr)
    arr = np.where(arr == 110, 0.05, arr)
    arr = np.where(arr == 120, 0.05, arr)
    arr = np.where(arr == 130, 0.1, arr)
    arr = np.where(arr == 140, 0.1, arr)
    arr = np.where(arr == 150, 0.1, arr)
    arr = np.where(arr == 160, 0.0, arr)
    arr = np.where(arr == 170, 0.0, arr)
    arr = np.where(arr == 180, 0.0, arr)
    arr = np.where(arr == 190, 0.0, arr)
    arr = np.where(arr == 200, 0.1, arr)
    arr = np.where(arr == 210, 0.0, arr)
    arr = np.where(arr == 220, 0.0, arr)
    arr = np.where(arr == 230, 0.0, arr)

    return arr
'''


def calc_final_suitability(elev_array, slope_array, prot_array, perm_array, lulc_array):
    """Calculate the suitability factor per gridcell where 0 is unsuitable and 1 is most suitable.

    """
    arr = elev_array * slope_array * prot_array * perm_array * lulc_array

    return arr


def get_hours_per_year(target_year) -> int:
    """Get the hours per week for leap and non-leap years based on the target year.

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


def calc_total_suitable_area_solar_PV(elev_raster, slope_raster, prot_raster, perm_raster, lulc_raster,
                                      output_directory,
                                      gridcellarea_raster):
    # create exclusion rasters for each exclusion category
    elev = process_elevation_solar(elev_raster)
    slope = process_slope_solar_PV(slope_raster)
    prot = process_protected(prot_raster)
    perm = process_permafrost_solar(perm_raster)
    lulc = process_lulc_solar(lulc_raster)
    gridcellarea = np.load(gridcellarea_raster)
    # replacing negative gridcell area values (-999.9) in ocean cells by 0.0
    gridcellarea = np.where(gridcellarea > 0.0, gridcellarea, 0.0)
    f_suit = calc_final_suitability(elev, slope, prot, perm, lulc)

    # calculate the suitable area in sqkm per gridcell (fi * ai)
    f_suit_sqkm = f_suit * gridcellarea[::-1, :]  # need to invert the lat index in gridcellarea

    out = os.path.join(output_directory, 'gridcellarea0p5deg.npy')
    np.save(out, gridcellarea)

    out = os.path.join(output_directory, 'solar_PV_suitability.npy')
    np.save(out, f_suit)

    out_suit_sqkm = os.path.join(output_directory, 'solar_PV_suitable_sqkm.npy')
    np.save(out_suit_sqkm, f_suit_sqkm)

    return f_suit_sqkm


def calc_total_suitable_area_solar_CSP(elev_raster, slope_raster, prot_raster, perm_raster, lulc_raster,
                                       output_directory,
                                       gridcellarea_raster):
    # create exclusion rasters for each exclusion category
    elev = process_elevation_solar(elev_raster)
    slope = process_slope_solar_CSP(slope_raster)
    prot = process_protected(prot_raster)
    perm = process_permafrost_solar(perm_raster)
    lulc = process_lulc_solar(lulc_raster)
    gridcellarea = np.load(gridcellarea_raster)
    # replacing negative gridcell area values (-999.9) in ocean cells by 0.0
    gridcellarea = np.where(gridcellarea > 0.0, gridcellarea, 0.0)
    f_suit = calc_final_suitability(elev, slope, prot, perm, lulc)

    # calculate the suitable area in sqkm per gridcell (fi * ai)
    f_suit_sqkm = f_suit * gridcellarea[::-1, :]  # need to invert the lat index in gridcellarea

    # out = os.path.join(output_directory, 'gridcellarea0p5deg.npy')
    # np.save(out, gridcellarea)

    out = os.path.join(output_directory, 'solar_CSP_suitability.npy')
    np.save(out, f_suit)

    out_suit_sqkm = os.path.join(output_directory, 'solar_CSP_suitable_sqkm.npy')
    np.save(out_suit_sqkm, f_suit_sqkm)

    return f_suit_sqkm


def calc_technical_potential_solar_PV(r_rsds, r_wind, r_tas, suit_sqkm_raster, yr_hours, output_directory, target_year):
    """Calculates the solar PV technical potential as an array in kWh per year.
       Reference: Gernaat et al. 2021; https://doi.org/10.1038/s41558-020-00949-9 """

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
    N_pv_daily = compute_solar_to_electric_eff(r_tas, r_rsds, r_wind)

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
    FLH = compute_FLH(solar_rad_KWh)

    # print FLH for debug
    FLH_raster = os.path.join(output_directory, 'FLH_' + str(target_year) + '.npy')
    np.save(FLH_raster, FLH)

    # computes the daily CSP_eff
    N_csp_daily = compute_CSP_eff(r_tas, r_rsds)

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
    wind_speed = power_law(wind, 125)  # Central based on Rinne et al. 2018
    # wind_speed = power_law(wind, 100) # Sensitivity based on Rinne et al. 2018
    # wind_speed = power_law(wind, 75)  # Sensitivity based on Rinne et al. 2018
    # wind_speed = power_law(wind, 150) # Sensitivity based on Rinne et al. 2018

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
    wadj = wind_speed_adjusted(wind_speed, rho_m)

    # Compute wind power at the hub height H using power curve from the representative turbine
    p_daily = wind_power_curve(wadj, wind_turbname)
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
