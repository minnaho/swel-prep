import sys
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyfuncs as pf
import scenario_style as ss

SCEN_KEYS = ['notidesnowec', 'ampwec', 'tidesnowec', 'tidesampwec']

data = {}
for name in SCEN_KEYS:
    data[name] = np.load(f'../postprocessing/wno3_env_{name}.npz', allow_pickle=True)

labels  = ss.LABELS
colors  = ss.COLORS
lstyles = ss.LSTYLES

axfont = 14

fig, (ax_ts, ax_pdf) = plt.subplots(2, 1, figsize=[12, 12])

for name in SCEN_KEYS:
    d = data[name]

    # Skipping first 12 hours for spin-up
    ts_min  = d['ts_min'][12:]
    ts_max  = d['ts_max'][12:]
    ts_avg  = d['ts_mean'][12:]

    bins  = d['bin_centers']
    pdf   = d['pdf']
    lbl   = labels[name]
    clr   = colors[name]
    ls    = lstyles[name]
    lw_ts  = ss.lw(name, base_lw=1.2)
    lw_pdf = ss.lw(name, base_lw=2.0)

    times = pf.numdate(d['ocean_times'][12:], 'seconds since 1995-01-01')

    # 1. Shaded area
    ax_ts.fill_between(times, ts_min, ts_max, color=clr, alpha=0.1, linewidth=0)

    # 2. Boundary lines
    ax_ts.plot(times, ts_max, color=clr, linestyle=ls, linewidth=lw_ts, label=lbl)
    ax_ts.plot(times, ts_min, color=clr, linestyle=ls, linewidth=lw_ts)

    # 3. PDF (PEAK NORMALIZATION)
    # Divide the PDF by its absolute max so the peak is at 1.0
    pdf_normalized = pdf / np.max(pdf)

    # Added label=lbl here so the legend works!
    ax_pdf.plot(bins, pdf_normalized, color=clr, linestyle=ls, linewidth=lw_pdf, label=lbl)

# Time series formatting
ax_ts.axhline(0, color='k', linewidth=0.8, alpha=0.5)
# Fixed the missing $ signs for math formatting here
ax_ts.set_ylabel(r"$w'NO_3'$ Envelope"
                 + "\n(mmol N m$^{-2}$ s$^{-1}$)", fontsize=axfont)
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax_ts.tick_params(axis='both', which='major', labelsize=axfont)
ax_ts.legend(fontsize=axfont, loc='upper left', ncol=2)

# PDF formatting
ax_pdf.set_yscale('log')
ax_pdf.set_xlabel(r"$w'NO_3'$ (mmol N m$^{-2}$ s$^{-1}$)", fontsize=axfont)
# Updated ylabel to reflect the visual normalization
ax_pdf.set_ylabel('Normalized PDF', fontsize=axfont)
ax_pdf.tick_params(axis='both', which='major', labelsize=axfont)
ax_pdf.legend(fontsize=axfont)

plt.tight_layout()
plt.savefig('./figs/wno3_flux_envelope.png', bbox_inches='tight', dpi=600)
print('saved ./figs/wno3_flux_envelope.png')
