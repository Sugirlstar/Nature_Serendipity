import cdsapi

c = cdsapi.Client()

for yr in range(1979, 2022):

    print(f'Downloading ERA5 Z500 for {yr}')
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'grid': 'F128',
            'gaussian': 'regular',
            'format': 'grib',
            'pressure_level': [
                '500',
            ],
            'year': [str(yr)],
            'month': [
                '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '06:00', '12:00', '18:00',
            ],
            'variable': 'geopotential',
        },
        f'/scratch/bell/hu1029/Data/raw/ERA5_Z500_F128/ERA5_Z500_6hr_{yr}.grb')
    print(f'Done downloading ERA5 Z500 for {yr} -------------------')
