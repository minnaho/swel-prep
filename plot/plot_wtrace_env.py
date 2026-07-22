import sys
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyfuncs as pf

# ==========================================
# SET YOUR TRACER HERE ('ptrace' or 'rtrace')
tracer = 'rtrace' 
# ==========================================

data = {}
for name in ['tides_wec', 'tides_nowec', 'notides_nowec', 'notides_wec']:
    # This will load either wptrace_env_... or wrtrace_env_...
    data[name] = np.load(f'../postprocessing/w{tracer}_env_{name}.npz', allow_pickle=True)

labels = {
    'tides_wec':     'WEC, tides',
    'notides_wec':   'WEC, no tides',
    'tides_nowec':   'no WEC, tides',
    'notides_nowec': 'no WEC, no tides',
}
colors = {
    'tides_wec':     'steelblue',
    'tides_nowec':   'navy',
    'notides_nowec': 'darkorange',
    'notides_wec':   'firebrick',
}
lstyles = {
    'tides_wec':     '-',
    'tides_nowec':   ':',
    'notides_nowec': '--',
    'notides_wec':   '-.',
}

axfont = 14

fig, (ax_ts, ax_pdf) = plt.subplots(2, 1, figsize=[12, 12]) 

# --- SCALING FACTORS ---
scale_ts  = 1e5
scale_pdf = 1e7

# Automatically calculate the exponent for the axis labels
ts_pow  = int(np.log10(scale_ts))
pdf_pow = int(np.log10(scale_pdf))

for name in ['tides_wec', 'tides_nowec', 'notides_nowec', 'notides_wec']:
    d = data[name]
    
    # Skipping first 12 hours for spin-up and applying the scales
    ts_min  = d['ts_min'][12:] * scale_ts
    ts_max  = d['ts_max'][12:] * scale_ts
    
    bins  = d['bin_centers'] * scale_pdf
    
    # --- PEAK NORMALIZATION ---
    # Get raw pdf (or counts) and divide by its absolute max so the peak is at 1.0
    pdf   = d['pdf'] 
    pdf_normalized = pdf / np.max(pdf)
    
    lbl   = labels[name]
    clr   = colors[name]
    ls    = lstyles[name]

    times = pf.numdate(d['ocean_times'][12:], 'seconds since 1995-01-01')

    # 1. Shaded area (translucent, no borders)
    ax_ts.fill_between(times, ts_min, ts_max, color=clr, alpha=0.1, linewidth=0)

    # 2. Boundary lines (using the specific linestyle)
    ax_ts.plot(times, ts_max, color=clr, linestyle=ls, linewidth=1.2, label=lbl)
    ax_ts.plot(times, ts_min, color=clr, linestyle=ls, linewidth=1.2)
    
    # 3. PDF (Using the normalized values)
    ax_pdf.plot(bins, pdf_normalized, color=clr, linestyle=ls, linewidth=2, label=lbl)

# --- Time series formatting ---
ax_ts.axhline(0, color='k', linewidth=0.8, alpha=0.5)

# Formatting the math text dynamically based on the tracer chosen
tracer_str = "ptrace" if tracer == 'ptrace' else "rtrace"

# Automatically uses the exponent calculated from scale_ts
ax_ts.set_ylabel(f"$w'{tracer_str}'$ Envelope\n($\\times 10^{{-{ts_pow}}}$ mmol tracer m$^{{-2}}$ s$^{{-1}}$)", fontsize=axfont)
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax_ts.tick_params(axis='both', which='major', labelsize=axfont)
ax_ts.legend(fontsize=axfont, loc='best', ncol=2)

# YOU WILL LIKELY NEED TO UNCOMMENT AND ADJUST THIS RANGE FOR THE NEW TRACERS
# ax_ts.set_ylim([-0.5, 0.5]) 

# --- PDF formatting ---
ax_pdf.set_yscale('log')

# Automatically uses the exponent calculated from scale_pdf
ax_pdf.set_xlabel(f"$w'{tracer_str}'$ ($\\times 10^{{-{pdf_pow}}}$ mmol tracer m$^{{-2}}$ s$^{{-1}}$)", fontsize=axfont)
ax_pdf.set_ylabel('Normalized PDF', fontsize=axfont)
ax_pdf.tick_params(axis='both', which='major', labelsize=axfont)
ax_pdf.legend(fontsize=axfont, loc='best')

plt.tight_layout()

# Save dynamically based on tracer
out_fig = f'./figs/w{tracer}_flux_envelope.png'
plt.savefig(out_fig, bbox_inches='tight', dpi=600)
print(f'saved {out_fig}')
