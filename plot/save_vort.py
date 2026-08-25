import sys
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
from netCDF4 import Dataset,num2date
import pyfuncs as pf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cmocean

grd = '/data/project3/minnaho/project9copy/swel/mc60_grd.nc'

grdnc = Dataset(grd,'r')
lat_nc = np.array(grdnc.variables['lat_rho'])
lon_nc = np.array(grdnc.variables['lon_rho'])-360
f_nc = np.array(grdnc.variables['f'])

# do this so land is masked in white
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc==0] = np.nan
# mask to contour
maskc = np.array(grdnc.variables['mask_rho'])


files1 = '/data/project3/minnaho/swel/notides/mc60/nowec/his/mc60_his.20190420230116.nc'
files2 = '/data/project3/minnaho/swel/notides/mc60/nowec/his/mc60_his.20190421110116.nc'

print('calculating urho, vrho')
wec_urho, wec_vrho = pf.rho_uv_angle(files1,grd,rotate=True)

nowec_urho, nowec_vrho = pf.rho_uv_angle(files2,grd,rotate=True)

print('calculating vorticity')
wec_vort = pf.vorticity(grd,wec_urho,wec_vrho)
nowec_vort = pf.vorticity(grd,nowec_urho,nowec_vrho)


wec_normvort = wec_vort/f_nc
nowec_normvort = nowec_vort/f_nc

oceantime = np.array(Dataset(files1,'r').variables['ocean_time'])
oceandt = pf.numdate(oceantime,'second since 1995-01-01')

oceantimenowec = np.array(Dataset(files2,'r').variables['ocean_time'])
oceandtnowec = pf.numdate(oceantimenowec,'second since 1995-01-01')

savewecnc = Dataset('notide_rossby'+files1[61:],'w',format='NETCDF4')
savewecnc.createDimension('time',wec_vort.shape[0])
savewecnc.createDimension('s_rho',wec_vort.shape[1])
savewecnc.createDimension('eta_rho',wec_vort.shape[2])
savewecnc.createDimension('xi_rho',wec_vort.shape[3])

wecwrite_time = savewecnc.createVariable('ocean_time','f8',('time'))
wecrelative_vort = savewecnc.createVariable('Rossby','f8',('time','s_rho','eta_rho','xi_rho'))

wecwrite_time[:] = oceantime
wecwrite_time.units = 'seconds since 1995-01-01'

wecrelative_vort[:] = wec_normvort
wecrelative_vort.units = ''
savewecnc.close()

savenowecnc = Dataset('notide_rossby'+files2[61:],'w',format='NETCDF4')
savenowecnc.createDimension('time',nowec_vort.shape[0])
savenowecnc.createDimension('s_rho',nowec_vort.shape[1])
savenowecnc.createDimension('eta_rho',nowec_vort.shape[2])
savenowecnc.createDimension('xi_rho',nowec_vort.shape[3])

nowecwrite_time = savenowecnc.createVariable('ocean_time','f8',('time'))
nowecrelative_vort = savenowecnc.createVariable('Rossby','f8',('time','s_rho','eta_rho','xi_rho'))

nowecwrite_time[:] = oceantimenowec
nowecwrite_time.units = 'seconds since 1995-01-01'

nowecrelative_vort[:] = nowec_normvort
nowecrelative_vort.units = ''
savenowecnc.close()


