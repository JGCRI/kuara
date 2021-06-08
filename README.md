# kuara
A geospatial package to estimate the technical and economic potential of renewable resources

# IN DEVELOPMENT

`kuara` is a Python package that computes the global solar (photovoltaic (PV) and concentrated solar power (CSP)) and wind onshore technical potentials at 0.5 geographic degree based on well-established methods described in the literature (Eurek et al. 2017; Gernaat et al. 2021; Karnauskas et al. 2018; Köberle et al. 2015; Rinne et al. 2018). `kuara` is the first open-source package for the purpose of estimation of renewable potentials with the advantage of allowing computations under distinct methodological assumptions. This enables researchers to assess the inherent uncertainty behind these estimates. The package accounts for a prebuilt methodology based on a literature survey. Specifically, the methods encompass the equations to compute the geographical and technical potentials and key assumptions pertaining to these equations, such as parameter values and choices of turbine technology in the case of wind power. Nevertheless, users have the flexibility to modify these assumptions as well as to replace or add individual components given the modular nature of the package.

## Getting Started with `kuara`

Set up Kuara using the following steps:

1. Install `kuara` from GitHub using:

```bash
fill in
```

2. Download the example data using the following in a Python prompt:

```python
import kuara

# the directory that you want to download and extract the example data to
data_dir = "<my data download location>"

# download and unzip the package data to your local machine
kuara.get_package_data(data_dir)
```

## Key functionality

Fill in
