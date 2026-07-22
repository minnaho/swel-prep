import sys
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

# Load PV PDFs
print('plotting')
wecpdf = np.load('../postprocessing/pdf7_pv_tideswec.npz')
nowecpdf = np.load('../postprocessing/pdf7_pv_tidesnowec.npz')
notidespdf = np.load('../postprocessing/pdf7_pv_notidesnowec.npz')

# Extract bins and counts
wecbin = wecpdf['bin_edges']
wecnum = wecpdf['pdf_wec']

nowecbin = nowecpdf['bin_edges']
nowecnum = nowecpdf['pdf_nowec']

notidesbin = notidespdf['bin_edges']
notidesnum = notidespdf['pdf_notidesnowec']

# Plot settings
figw = 12
figh = 8
axisfont = 16

fig, ax = plt.subplots(1, 1, figsize=[figw, figh])

# Plot data
# Calculate bin centers (length 500)
wec_centers = (wecbin[:-1] + wecbin[1:]) / 2

# Plot against centers, and remove the appended 0 from your PDF arrays
ax.plot(wec_centers, wecnum[:-1], color='lightblue', linewidth=1.5, label='WEC, tides')
ax.plot(wec_centers, nowecnum[:-1], color='navy', linestyle=':', linewidth=1.5, label='no WEC, tides')
ax.plot(wec_centers, notidesnum[:-1], color='orange', linestyle='--', linewidth=1.5, label='no WEC, no tides')

# Axis scales and limits
ax.set_yscale('log')
ax.set_ylim([1E-5, 1E-1])

# IMPORTANT: Adjust this xlim to match whatever bin_edges you used in the extraction script!
ax.set_xlim([-1e-7, 1e-7])

ax.grid(True, which="major", axis="y", color='gray', linestyle='-', alpha=0.3)
ax.grid(True, which="minor", axis="y", color='gray', linestyle=':', alpha=0.2)

# Labels and ticks
ax.legend(fontsize=axisfont)
ax.set_xlabel('Potential Vorticity (s$^{-3}$)', fontsize=axisfont)
ax.set_ylabel('PDF', fontsize=axisfont)
ax.tick_params(axis='both', which='major', labelsize=axisfont)

# Force scientific notation on the x-axis so it doesn't get cluttered with zeros
ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
ax.xaxis.get_offset_text().set_fontsize(axisfont)

# Save figure
plt.savefig('./figs/pdf_pv7.png', bbox_inches='tight')
print('Saved figure to ./figs/pdf_pv7.png')
