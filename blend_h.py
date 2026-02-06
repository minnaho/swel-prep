# IMPORTANT - unused since I found a better DEM product that 
# encompasses entire grid
# https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ngdc.mgg.dem:3545/html
#----------------------------------
# blend SRTM 1.5 bathymetry offshore to 
# Monterey Bay DEM product
# https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ngdc.mgg.dem:monterey_bay_P080_2018
#----------------------------------

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interp1d

plt.ion()

pgrd = Dataset('mc60_grd.nc','r')
cgrd = Dataset('mc60_new_grd.nc','r+')

pmask = np.array(pgrd.variables['mask_rho'])
cmask = np.array(cgrd.variables['mask_rho'])

#mask_new = np.copy(pmask)

pbathy = np.array(pgrd.variables['h'])
cbathy = np.array(cgrd.variables['h'])

chj, chi = np.where(np.isnan(cbathy))

# find where "seam" between old and new h is 
linej_list = [] 
linei_list = [] 

for j_j in range(280,880): 
    for i_i in range(350,500): 
        if np.isfinite(cbathy[j_j,i_i]): 
            linej_list.append(j_j) 
            linei_list.append(i_i) 
            break 

linej = np.array(linej_list) 
linei = np.array(linei_list)


# blend parent and child, prioritizing child
blend_width = 16

hnew = np.copy(pbathy)
ny, nx = pbathy.shape

for j in range(ny):

    ii_valid = np.where(np.isfinite(cbathy[j, :]))[0]
    if ii_valid.size == 0:
        continue

    # seam index (first child cell)
    i0 = ii_valid.min()

    imin = max(i0 - blend_width, 0)
    imax = i0 + 1

    di = np.arange(imin, imax) - i0
    xi = di / blend_width

    # cosine taper: parent → child
    w = 0.5 * (1 + np.cos(np.pi * xi))
    w[xi <= -1] = 0.0
    w[xi >=  0] = 1.0

    # child value AT THE SEAM
    cval = cbathy[j, i0]
    #nearest cbathy value, not just i0
    #cval = np.nanmean(cbathy[j, ii_valid[:3]])


    for ii, wi in zip(range(imin, imax), w):

        #if mask_new[j, ii] == 0:
        #    continue

        hnew[j, ii] = (
            wi * cval +
            (1.0 - wi) * pbathy[j, ii]
        )

# child wins inside its domain
child_mask = np.isfinite(cbathy)
hnew[child_mask] = cbathy[child_mask]

# enforce land
#hnew[mask_new == 0] = 0

# start from blended bathymetry
hnew_masked = np.copy(hnew)

ny, nx = cbathy.shape

'''
# interpolate seam column to all rows if needed
f = interp1d(linej, linei, kind='linear', bounds_error=False, fill_value='extrapolate')
linei_all = f(np.arange(ny)).astype(int)

# start from blended bathymetry
for j, ii in enumerate(linei_all):
    ii = int(np.clip(ii, 0, nx-1))  # prevent out-of-bounds

    # --- Right of seam ---
    right_cols = np.arange(ii+1, nx)
    nan_right = right_cols[np.isnan(cbathy[j, right_cols])]
    hnew_masked[j, nan_right] = 0

    # --- South of seam ---
    south_rows = np.arange(j+1, ny)
    nan_south = south_rows[np.isnan(cbathy[south_rows, ii])]
    hnew_masked[nan_south, ii] = 0

    # --- North of seam ---
    north_rows = np.arange(0, j)
    nan_north = north_rows[np.isnan(cbathy[north_rows, ii])]
    hnew_masked[nan_north, ii] = 0
'''

hnew_masked = np.copy(hnew)
ny, nx = cbathy.shape

# interpolate seam column to all rows
from scipy.interpolate import interp1d
f = interp1d(linej, linei, kind='linear', bounds_error=False, fill_value='extrapolate')
linei_all = f(np.arange(ny)).astype(int)

for j, ii in enumerate(linei_all):
    ii = int(np.clip(ii, 0, nx-1))

    # --- Right of seam ---
    right_cols = np.arange(ii+1, nx)
    right_nan = np.isnan(cbathy[j, right_cols])
    hnew_masked[j, right_cols[right_nan]] = 0

    # --- South of seam ---
    #south_rows = np.arange(j+1, ny)
    ## Only set if cbathy[south_row, ii] is NaN (child patch)
    #south_nan = np.isnan(cbathy[south_rows, ii])
    #hnew_masked[south_rows[south_nan], ii] = 0

    # --- North of seam ---
    north_rows = np.arange(0, j)
    north_nan = np.isnan(cbathy[north_rows, ii])
    hnew_masked[north_rows[north_nan], ii] = 0

hardcut = np.copy(cbathy)
hardcut[chj,chi] = pbathy[chj,chi]

fig1,ax1 = plt.subplots(1,1)
cplt = ax1.imshow(cbathy,origin='lower',cmap='rainbow',vmin=50,vmax=830)
fig1.colorbar(cplt)
ax1.set_title('Monterey product')
lplt = ax1.plot(linei,linej,color='k',linewidth=3)

fig2,ax2 = plt.subplots(1,1)
pplt = ax2.imshow(pbathy,origin='lower',cmap='rainbow',vmin=50,vmax=830)
fig2.colorbar(pplt)
ax2.set_title('old Monterey grid')

fig3,ax3 = plt.subplots(1,1)
hplt = ax3.imshow(hnew,origin='lower',cmap='rainbow',vmin=100,vmax=830)
fig3.colorbar(hplt)
ax3.set_title('blended Monterey grid')

fig4,ax4 = plt.subplots(1,1)
hplt = ax4.imshow(hardcut,origin='lower',cmap='rainbow',vmin=100,vmax=830)
fig4.colorbar(hplt)
ax4.set_title('hardcut Monterey grid')



fig5,ax5 = plt.subplots(1,1)
hplt = ax5.imshow(hnew,origin='lower',cmap='rainbow',vmin=0,vmax=100)
fig5.colorbar(hplt)
ax5.set_title('blended Monterey grid')

fig6,ax6 = plt.subplots(1,1)
hmplt = ax6.imshow(hnew_masked,origin='lower',cmap='rainbow',vmin=0,vmax=100)
fig6.colorbar(hmplt)
ax6.set_title('blended Monterey grid with masking')
lplt = ax6.plot(linei,linej,color='k',linewidth=3)

# fix mask
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# assume hnew_masked already exists
hnew_manual = np.copy(hnew_masked)
hnew_manual[hnew_manual<=2] = 0

# Plot for interactive selection
fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.imshow(hnew_manual, origin='lower', cmap='viridis',vmin=0,vmax=100)
fig.colorbar(cax)
ax.set_title('Click points or drag rectangles to set to 0 (land)')

# --- Rectangle selection ---
def onselect(eclick, erelease):
    x0, y0 = int(round(eclick.xdata)), int(round(eclick.ydata))
    x1, y1 = int(round(erelease.xdata)), int(round(erelease.ydata))
    # sort indices
    j0, j1 = sorted([y0, y1])
    i0, i1 = sorted([x0, x1])
    hnew_manual[j0:j1+1, i0:i1+1] = 0
    # update plot
    cax.set_data(hnew_manual)
    fig.canvas.draw_idle()

rect_sel = RectangleSelector(ax, onselect, drawtype='box', useblit=True,
                             button=[1], minspanx=1, minspany=1, spancoords='pixels')

# --- Point selection ---
clicked_points = []

def onclick(event):
    if event.inaxes != ax:
        return
    i = int(round(event.xdata))
    j = int(round(event.ydata))
    hnew_manual[j, i] = 0
    clicked_points.append((j, i))
    cax.set_data(hnew_manual)
    fig.canvas.draw_idle()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

# --- Finish interaction ---
print("Instructions:")
print(" - Left click: mark single point as 0")
print(" - Click+drag: draw rectangle to mark area as 0")
print(" - Close the figure window when done")

plt.show()

# After closing the figure, hnew_manual contains the corrected bathymetry
print(f"Total points manually set: {len(clicked_points)}")

# hnew_manual can now be used or saved back to NetCDF
newmask = np.copy(hnew_manual)
newmask[hnew_manual>0] = 1
