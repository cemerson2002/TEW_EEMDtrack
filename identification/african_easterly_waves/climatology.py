import numpy as np
import xarray as xr
import metpy.calc as mpcalc


def smthClmDay(clmDay, nHarm):  # from athul
    # nharm is the number of harmonic modes that needs to be removed. So we retain that in smthClm to later 
    # subtract from raw.

    cf = np.fft.rfft(clmDay, axis=0)  # ([nday/2]+1)x[nlat]x[mlon]<==n:even
    # ([nday+1/2])x[nlat]x[mlon]<==n:odd
    # cf(:,0:nharm-1)
    cf[nHarm, :, :] = 0.5 * cf[nHarm, :, :]  # mini-taper
    cf[nHarm + 1:, :, :] = 0.0  # set all higher coef. to zero

    clmDaySmth = np.fft.irfft(cf, axis=0)  # reverse FT to real reconstruct data

    clmDaySmth = xr.DataArray(clmDaySmth, dims=('hourofyear', 'lat', 'lon'),
                              coords={'hourofyear': clmDay.hourofyear, 'lat': clmDay.lat, 'lon': clmDay.lon})
    return (clmDaySmth)

def compute_climatology(cfg):

    cv_database_dir = cfg.cv_data_dir

    f = xr.open_dataset(cv_database_dir + "aew_input.nc")
    # smoothing
    cv = mpcalc.smooth_gaussian(f.cv, n=13).metpy.dequantify()
    cv_smooth = cv

#    # generate hourofyear climatology, save it to file for next time
#
#    # Calculate a hour-of-year climatology to be subtracted later.
#    cv = cv.assign_coords(hourofyear=cv.time.dt.strftime("%m-%d %H"))
#    print('Computing climatology...')
#    cv_clim_da = cv.groupby('hourofyear').mean('time')
#
#    # print('Smoothing climatology...')
#    # clm = smthClmDay(clm, 3)  # calculate climatology with just the 3 harmonic modes retained.
#    print('Climatology computed ')
#
#    cv_clim_da.attrs['units'] = 's**-1'
#    cv_clim_da.attrs['standard_name'] = 'Curvature Vorticity'
#    cv_clim_da.attrs['long_name'] = 'Curvature Vorticity climatology (hour of year)'
#
#    cv_clim = cv_clim_da.to_dataset(name="cv")
    cv_clim = cv_smooth.to_dataset(name='cv')
    
    print('Computing per-latitude CV standard deviation...')
    cv_sigma = cv.std(dim=['time','lon'])
    cv_sigma.attrs['units'] = 's**-1'
    cv_sigma.attrs['standard_name'] = 'Curvature Vorticity STD'
    cv_clim['cv_sigma'] = cv_sigma

#    # compute anomalies
#    full_cv_anomalies = cv.groupby('hourofyear') - cv_clim.cv
#
#    # quantile applied over time,lat,lon -> so one scalar of cfg.cv_percentile% quantile per pressure level
#    quantiles_full = full_cv_anomalies.quantile(float(cfg.cv_percentile) / 100.0, dim=['lat', 'lon', 'time'])
#    cv_clim['cva_quantile_full'] = quantiles_full
#
#    quantiles_hoy = full_cv_anomalies.groupby('hourofyear').quantile(
#        q = float(cfg.cv_percentile) / 100
#        )
#
#    cv_clim['cva_quantile_hoy'] = quantiles_hoy
    
    print("Done!")
    print("Generated dataset:")
    print(cv_clim)

    return cv_clim

