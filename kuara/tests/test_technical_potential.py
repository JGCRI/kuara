import unittest

import kuara.technical_potential as tp


class TestTechnicalPotential(unittest.TestCase):

    def test_power_law(self):

        result = tp.power_law(7, 100)

        self.assertEqual(9.726468460611963, result)

    def test_dry_air_density_ideal(self):

        result = tp.dry_air_density_ideal(7.77, 7.77)

        self.assertEqual(0.003483616551358959, result)

    def test_dry_air_density_humidity(self):

        result = tp.dry_air_density_humidity(7.77, 7.77)

        self.assertEqual(5.046900702344034, result)


    def test_wind_speed_adjusted(self):

        result = tp.wind_speed_adjusted(7.77, 7.77)

        self.assertEqual(14.382994978685954, result)

    def test_power_curve_vestas_v136_3450(self):

        pass

    def test_power_curve_vestas_v90_2000(self):

        pass

    def test_power_curve_GE_2500(self):

        pass

    def test_power_curve_E101_3050(self):

        pass

    def test_power_curve_Gamesa_G114_2000(self):

        pass

    def test_power_curve_GE1500(self):

        pass

    def test_power_curve_IEC_classII_3500(self):

        pass

    def test_compute_solar_to_electric_eff(self):

        pass

    def test_compute_FLH(self):

        pass

    def test_compute_CSP_eff(self):

        pass

    def test_read_climate_data(self):

        pass

    def test_process_climate_solar(self):

        pass

    def test_process_climate(self):

        pass

    def test_process_elevation(self):

        pass

    def test_process_slope(self):

        pass

    def test_process_protected(self):

        pass

    def test_process_permafrost(self):

        pass

    def test_process_lulc(self):

        pass

    def test_process_elevation_solar(self):

        pass

    def test_process_slope_solar_PV(self):

        pass

    def test_process_slope_solar_CSP(self):

        pass

    def test_process_permafrost_solar(self):

        pass

    def test_process_lulc_solar(self):

        pass

    def test_calc_final_suitability(self):

        pass

    def test_get_hours_per_year(self):

        no_leap = tp.get_hours_per_year(2019)
        leap = tp.get_hours_per_year(2020)

        self.assertEqual(8760, no_leap)
        self.assertEqual(8784, leap)

    def test_calc_total_suitable_area(self):

        pass

    def test_calc_total_suitable_area_solar_PV(self):

        pass

    def test_calc_total_suitable_area_solar_CSP(self):

        pass

    def test_calc_technical_potential_solar_PV(self):

        pass

    def test_calc_technical_potential_solar_CSP(self):

        pass

    def test_calc_technical_potential(self):

        pass


if __name__ == '__main__':
    unittest.main()
