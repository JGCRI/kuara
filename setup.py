from setuptools import setup, find_packages


setup(
    name='kuara',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/JGCRI/kuara',
    license='BSD 2-Clause',
    author='Chris R. Vernon; Silvia R. Silva da Santos',
    author_email='chrisrvernon@gmail.com',
    description='A geospatial package to estimate the technical and economic potential of renewable resources',
    python_requires='>=3.6',
    install_requires=[
        'rasterio~=1.2.3',
        'numpy~=1.20.3',
        'xarray~=0.18.2',
        'netCDF4~=1.5.6',
        'scipy~=1.6.3'
    ],
    include_package_data=True
)