import sys
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyfuncs as pf
import scenario_style as ss

# ==========================================
# SET YOUR TRACER HERE ('ptrace' or 'rtrace')
tracer = 'rtrace'
# ==========================================

SCEN_KEYS = ['notidesnowec', 'ampwec', 'tidesnowec', 'tidesampwec']

# (npz filename suffix, depth label, output filename suffix) -- matches
# calc_wtrace_flux.py's DEPTHS, which writes w{tracer}_env_{name}.npz (10m)
# and w{tracer}_env_20m_{name}.npz (20m)
DEPTHS = [('', '10 m', ''), ('_20m', '20 m', '_20m')]

labels  = ss.LABELS
colors  = ss.COLORS
lstyles = ss.LSTYLES

axfont = 14

# --- SCALING FACTORS ---
scale_ts  = 1e5
scale_pdf = 1e7

# Automatically calculate the exponent for the axis labels
ts_pow  = int(np.log10(scale_ts))
pdf_pow = int(np.log10(scale_pdf))

# Formatting the math text dynamically based on the tracer chosen
tracer_str = "ptrace" if tracer == 'ptrace' else "rtrace"

for npz_suffix, depth_label, out_suffix in DEPTHS:
    data = {}
    for name in SCEN_KEYS:
        # This will load either wptrace_env_... or wrtrace_env_..., at 10m or 20m
        data[name] = np.load(f'../postprocessing/w{tracer}_env{npz_suffix}_{name}.npz', allow_pickle=True)

    fig, (ax_ts, ax_pdf) = plt.subplots(2, 1, figsize=[12, 12])

    for name in SCEN_KEYS:
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
        lw_ts  = ss.lw(name, base_lw=1.2)
        lw_pdf = ss.lw(name, base_lw=2.0)

        times = pf.numdate(d['ocean_times'][12:], 'seconds since 1995-01-01')

        # 1. Shaded area (translucent, no borders)
        ax_ts.fill_between(times, ts_min, ts_max, color=clr, alpha=0.1, linewidth=0)

        # 2. Boundary lines (using the specific linestyle)
        ax_ts.plot(times, ts_max, color=clr, linestyle=ls, linewidth=lw_ts, label=lbl)
        ax_ts.plot(times, ts_min, color=clr, linestyle=ls, linewidth=lw_ts)

        # 3. PDF (Using the normalized values)
        ax_pdf.plot(bins, pdf_normalized, color=clr, linestyle=ls, linewidth=lw_pdf, label=lbl)

    # --- Time series formatting ---
    ax_ts.axhline(0, color='k', linewidth=0.8, alpha=0.5)

    # Automatically uses the exponent calculated from scale_ts
    ax_ts.set_ylabel(f"$w'{tracer_str}'$ Envelope ({depth_label})\n($\\times 10^{{-{ts_pow}}}$ mmol tracer m$^{{-2}}$ s$^{{-1}}$)", fontsize=axfont)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax_ts.tick_params(axis='both', which='major', labelsize=axfont)
    ax_ts.legend(fontsize=axfont, loc='best', ncol=2)

    # YOU WILL LIKELY NEED TO UNCOMMENT AND ADJUST THIS RANGE FOR THE NEW TRACERS
    # ax_ts.set_ylim([-0.5, 0.5])

    # --- PDF formatting ---
    ax_pdf.set_yscale('log')

    # Automatically uses the exponent calculated from scale_pdf
    ax_pdf.set_xlabel(f"$w'{tracer_str}'$ ({depth_label}) ($\\times 10^{{-{pdf_pow}}}$ mmol tracer m$^{{-2}}$ s$^{{-1}}$)", fontsize=axfont)
    ax_pdf.set_ylabel('Normalized PDF', fontsize=axfont)
    ax_pdf.tick_params(axis='both', which='major', labelsize=axfont)
    ax_pdf.legend(fontsize=axfont, loc='best')

    plt.tight_layout()

    # Save dynamically based on tracer and depth
    out_fig = f'./figs/w{tracer}_flux_envelope{out_suffix}.png'
    plt.savefig(out_fig, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'saved {out_fig}')
