"""Canonical per-scenario line styling (color / linestyle / label / linewidth).
Import this instead of redefining COLORS/LSTYLES/LABELS in each plot script."""

STYLE = {
    'tideswec':      dict(color='steelblue', ls='-',  label='tides, WEC'),
    'tidesnowec':    dict(color='navy',      ls=':',  label='tides, no WEC'),
    'notidesnowec':  dict(color='darkorange', ls='--', label='no tides, no WEC'),
    'notideswec':    dict(color='firebrick', ls='-.', label='no tides, WEC'),
    # ampwec (raw naming) and notidesampwec (zslice-dir naming) are the SAME run
    'ampwec':        dict(color='pink',     ls='--', label='no tides, 2.5× WEC'),
    'notidesampwec': dict(color='pink',     ls='--', label='no tides, 2.5× WEC'),
    'tidesampwec':   dict(color='cyan',      ls='--', label='tides, 2.5× WEC'),
    # smode200 parent solution (no WEC) -- neutral tones so these read as
    # reference curves and can't be confused with a child mc60 scenario
    'parent_tides':   dict(color='black',   ls='-',  label='parent, tides'),
    'parent_notides': dict(color='dimgray', ls='-.', label='parent, no tides'),
}

# amplified-WEC runs get the thicker line
AMP_SCENS = {'ampwec', 'notidesampwec', 'tidesampwec'}

# underscore key spelling used by plot_wno3_flux.py -> canonical key
ALIASES = {
    'tides_wec': 'tideswec', 'tides_nowec': 'tidesnowec',
    'notides_nowec': 'notidesnowec', 'notides_wec': 'notideswec',
}

def _r(scen):        return ALIASES.get(scen, scen)
def color(scen):     return STYLE[_r(scen)]['color']
def ls(scen):        return STYLE[_r(scen)]['ls']
def label(scen):     return STYLE[_r(scen)]['label']
def lw(scen, base_lw=2.0):
    return base_lw + 1.5 if _r(scen) in AMP_SCENS else base_lw

def line_kwargs(scen, base_lw=2.0, **overrides):
    """Ready-to-splat kwargs for ax.plot/loglog: color, linestyle, linewidth, label."""
    r = _r(scen)
    kw = dict(color=STYLE[r]['color'], linestyle=STYLE[r]['ls'],
              linewidth=lw(scen, base_lw), label=STYLE[r]['label'])
    kw.update(overrides)
    return kw

# drop-in derived dicts for scripts that index COLORS[scen] etc. directly
COLORS  = {s: v['color'] for s, v in STYLE.items()}
LSTYLES = {s: v['ls']    for s, v in STYLE.items()}
LABELS  = {s: v['label'] for s, v in STYLE.items()}
