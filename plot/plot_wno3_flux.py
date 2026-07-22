import sys
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import pyfuncs as pf
import scenario_style as ss

SCENARIOS = ['tides_wec', 'tides_nowec', 'notides_nowec', 'notides_wec']

labels  = {n: ss.label(n) for n in SCENARIOS}
colors  = {n: ss.color(n) for n in SCENARIOS}
lstyles = {n: ss.ls(n)    for n in SCENARIOS}

# ── per-tracer config ─────────────────────────────────────────────────────────
# ts_npz  : NPZ with the time series (key 'time_series' or 'ts_mean')
# ts_key  : key name for the mean time series
# env_npz : NPZ with ts_min / ts_max / bin_centers / pdf
# scale_ts  : multiply time series values for readability
# scale_pdf : multiply bin_centers for readability
# units   : physical unit string for axis labels
# math    : LaTeX math symbol for the variable

TRACERS = {
    'NO3': dict(
        ts_npz   = lambda n: f'../postprocessing/wno3_flux_{n}.npz',
        ts_key   = 'time_series',
        env_npz  = lambda n: f'../postprocessing/wno3_env_{n}.npz',
        scale_ts  = 1.0,
        scale_pdf = 1.0,
        units    = r'mmol N m$^{-2}$ s$^{-1}$',
        math     = r"$w'NO_3'$",
        outfile  = './figs/wno3_flux.png',
    ),
    'ptrace': dict(
        ts_npz   = lambda n: f'../postprocessing/wptrace_env_{n}.npz',
        ts_key   = 'ts_mean',
        env_npz  = lambda n: f'../postprocessing/wptrace_env_{n}.npz',
        scale_ts  = 1e5,
        scale_pdf = 1e7,
        units    = r'mmol m$^{-2}$ s$^{-1}$',
        math     = r"$w'ptrace'$",
        outfile  = './figs/wptrace_flux.png',
    ),
    'rtrace': dict(
        ts_npz   = lambda n: f'../postprocessing/wrtrace_env_{n}.npz',
        ts_key   = 'ts_mean',
        env_npz  = lambda n: f'../postprocessing/wrtrace_env_{n}.npz',
        scale_ts  = 1e5,
        scale_pdf = 1e7,
        units    = r'mmol m$^{-2}$ s$^{-1}$',
        math     = r"$w'rtrace'$",
        outfile  = './figs/wrtrace_flux.png',
    ),
}

axfont = 16

for tracer, cfg in TRACERS.items():
    sc  = cfg['scale_ts']
    scp = cfg['scale_pdf']
    ts_pow  = int(np.log10(sc))  if sc  != 1.0 else None
    pdf_pow = int(np.log10(scp)) if scp != 1.0 else None

    ts_ylabel  = (f"{cfg['math']} mean\n"
                  + (f"($\\times 10^{{-{ts_pow}}}$ " if ts_pow else '(')
                  + f"{cfg['units']})")
    env_ylabel = (f"{cfg['math']} envelope\n"
                  + (f"($\\times 10^{{-{ts_pow}}}$ " if ts_pow else '(')
                  + f"{cfg['units']})")
    pdf_xlabel = (f"{cfg['math']} "
                  + (f"($\\times 10^{{-{pdf_pow}}}$ " if pdf_pow else '(')
                  + f"{cfg['units']})")

    fig = plt.figure(figsize=[14, 14])
    gs  = gridspec.GridSpec(3, 2, width_ratios=[4, 1], hspace=0.35, wspace=0.08,
                            figure=fig)
    ax_ts  = fig.add_subplot(gs[0, 0])
    ax_box = fig.add_subplot(gs[0, 1], sharey=ax_ts)
    ax_env = fig.add_subplot(gs[1, :])
    ax_pdf = fig.add_subplot(gs[2, :])

    ts_data = {}   # collect for box plot

    for name in SCENARIOS:
        d_ts  = np.load(cfg['ts_npz'](name),  allow_pickle=True)
        d_env = np.load(cfg['env_npz'](name), allow_pickle=True)
        lbl   = labels[name]
        clr   = colors[name]
        ls    = lstyles[name]

        # time series — skip first 12 steps (spin-up / notides_wec artifact)
        ts    = d_ts[cfg['ts_key']][12:] * sc
        times = pf.numdate(d_ts['ocean_times'][12:], 'seconds since 1995-01-01')
        ax_ts.plot(times, ts, color=clr, linestyle=ls, linewidth=1.5, label=lbl)
        ts_data[name] = ts

        # envelope
        t_env   = pf.numdate(d_env['ocean_times'][12:], 'seconds since 1995-01-01')
        ts_min  = d_env['ts_min'][12:] * sc
        ts_max  = d_env['ts_max'][12:] * sc
        ax_env.fill_between(t_env, ts_min, ts_max, color=clr, alpha=0.1, linewidth=0)
        ax_env.plot(t_env, ts_max, color=clr, linestyle=ls, linewidth=1.2, label=lbl)
        ax_env.plot(t_env, ts_min, color=clr, linestyle=ls, linewidth=1.2)

        # PDF — peak normalization
        pdf_norm = d_env['pdf'] / np.max(d_env['pdf'])
        ax_pdf.plot(d_env['bin_centers'] * scp, pdf_norm,
                    color=clr, linestyle=ls, linewidth=1.5, label=lbl)

    # box and whisker
    short_labels = [labels[n].replace(', ', '\n') for n in SCENARIOS]
    for i, name in enumerate(SCENARIOS):
        clr = colors[name]
        bp = ax_box.boxplot(ts_data[name], positions=[i], widths=0.55,
                            patch_artist=True, showmeans=True,
                            meanprops={'marker': 'o', 'markerfacecolor': clr,
                                       'markeredgecolor': 'k', 'markersize': 6},
                            medianprops={'color': 'k', 'linewidth': 1.5},
                            boxprops={'facecolor': clr, 'alpha': 0.5},
                            whiskerprops={'color': clr, 'linewidth': 1.2},
                            capprops={'color': clr, 'linewidth': 1.2},
                            flierprops={'marker': '.', 'markerfacecolor': clr,
                                        'markersize': 2, 'alpha': 0.3})
    ax_box.set_xticks(range(len(SCENARIOS)))
    ax_box.set_xticklabels(short_labels, fontsize=8)
    ax_box.axhline(0, color='k', linewidth=0.5)
    ax_box.tick_params(axis='y', labelleft=False)
    ax_box.tick_params(axis='x', which='major', labelsize=8)

    # time series formatting
    ax_ts.axhline(0, color='k', linewidth=0.5)
    ax_ts.set_ylabel(ts_ylabel, fontsize=axfont)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax_ts.tick_params(axis='both', which='major', labelsize=axfont)

    # envelope formatting
    ax_env.axhline(0, color='k', linewidth=0.5)
    ax_env.set_ylabel(env_ylabel, fontsize=axfont)
    ax_env.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax_env.tick_params(axis='both', which='major', labelsize=axfont)

    # PDF formatting
    ax_pdf.set_yscale('log')
    ax_pdf.set_xlabel(pdf_xlabel, fontsize=axfont)
    ax_pdf.set_ylabel('Normalized PDF', fontsize=axfont)
    ax_pdf.tick_params(axis='both', which='major', labelsize=axfont)
    ax_pdf.legend(loc='upper right',fontsize=axfont)

    plt.savefig(cfg['outfile'], bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f"saved {cfg['outfile']}")
