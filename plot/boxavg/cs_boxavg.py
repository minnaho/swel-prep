"""Perpendicular-box spatial averaging for the plot_cs_*_box.py family.

Each plot_cs_*.py script in ../ samples a single line through the grid (a
diagonal transect via map_coordinates, or a fixed-eta row). The _box.py
copies in this directory instead average over a box straddling that line,
so each section shows the mean structure across a band rather than one
line of 60 m grid cells.

mc60 is 60 m isotropic (dx/dy = 1.0000000002, confirmed against mc60_grd.nc
pm/pn), so a perpendicular offset in (eta, xi) index space is a true
physical perpendicular offset, and one index step = 60 m. No metric
correction is needed.

Averaging is done on FIXED DEPTH LEVELS, not s-levels: across a 1.2 km box
(HALF_WIDTH_PTS=10) the seafloor depth varies by ~22% (median), up to ~126%
(p90), on the canyon-crossing mid transect (eta=477) -- the same s_rho
index sits at very different real depths across the box. build_box_transect
/ build_box_mid_transect attach a common 'zgrid' to each transect; every
offset column is interpolated onto that grid (interp_to_z) before being
averaged (boxavg_section). Cells below the local seafloor / above the local
surface fall outside the column's z-range and become NaN, so they drop out
of the average at that depth rather than polluting it.

The two zsliced *_diff.py scripts already sit on a fixed depth grid at
every time step, so they use boxavg_section_fixedz instead: no vertical
interpolation, just an average across offsets at each existing depth level.

PERFORMANCE NOTE: interp_section/interp_transect used to call
scipy.ndimage.map_coordinates once per depth level per offset, but profiling
showed map_coordinates itself is <1% of that cost -- the other 99% was the
np.isnan/np.where/astype full-DOMAIN (1202x702) scan done every call just to
sample the ~300 transect points. build_box_transect now precomputes, once
per offset, the 4 clamped bilinear-neighbour indices and weights
(_bilinear_gather); interp_section then does a single vectorized gather
across all depth levels at once instead of looping. This reproduces
map_coordinates(order=1, mode='nearest') bit-for-bit up to float summation
order (verified empirically: NaN pattern identical, max abs diff ~1e-15 on
realistic transect geometry) -- see the equivalence self-test below.
"""

import os
import numpy as np
from scipy.ndimage import map_coordinates

# ---------------------------------------------------------------------------
# Path anchoring -- the _box.py scripts live one level below plot/, so they
# still read plot/mc60_grd.nc (no grid file duplicated here), but write to
# their own boxavg/figs/ tree rather than plot/figs/, so box-averaged output
# never mixes with the single-line originals'.
# ---------------------------------------------------------------------------
HERE     = os.path.dirname(os.path.abspath(__file__))   # .../plot/boxavg
PLOT_DIR = os.path.dirname(HERE)                        # .../plot
GRD      = os.path.join(PLOT_DIR, 'mc60_grd.nc')


def figpath(*parts):
    """.../plot/boxavg/figs/<parts>, creating no directories itself --
    callers still need os.makedirs on the parent for new subdirectories."""
    return os.path.join(HERE, 'figs', *parts)


# ---------------------------------------------------------------------------
# Box geometry
# ---------------------------------------------------------------------------
HALF_WIDTH_PTS = 10     # grid points either side of the line (600 m at 60 m/cell)
DZ             = 2.0    # z-grid spacing (m) for the box average
MIN_FRAC       = 0.0    # require this fraction of the box valid at a cell, else NaN


def box_offsets(half_width=HALF_WIDTH_PTS):
    """Integer offsets [-half_width .. +half_width] across the box."""
    return np.arange(-half_width, half_width + 1)


def perp_unit(eta, xi):
    """Unit vector perpendicular to the line defined by eta[0]->eta[-1],
    xi[0]->xi[-1], in (eta, xi) index space. Since mc60 is isotropic this
    is also the true physical perpendicular direction.
    ts (eta0=271,xi0=543,slope=-0.8) -> (-0.781, -0.625)
    tn (eta0=832,xi0=646,slope=0.6)  -> (-0.857,  0.514)"""
    de = eta[-1] - eta[0]
    dx = xi[-1] - xi[0]
    L = np.hypot(de, dx)
    return dx / L, -de / L


def make_zgrid(depth_lim, dz=DZ):
    """Ascending z levels from <= depth_lim up to and including 0, spaced by
    dz. Built from 0 downward (rather than np.arange(depth_lim, dz, dz)) so
    it always includes 0 exactly regardless of depth_lim's parity."""
    n = int(np.ceil(-depth_lim / dz))
    return -dz * np.arange(n, -1, -1)


def _bilinear_gather(coords, shape):
    """Precompute clamped bilinear-neighbour indices/weights for `coords`
    (2, n_pts) on a grid of `shape` (ny, nx), reproducing
    scipy.ndimage.map_coordinates(..., order=1, mode='nearest') exactly:
    'nearest' boundary handling is equivalent to clamping BOTH neighbour
    indices to [0, ny-1]/[0, nx-1] before the linear-interpolation weights
    are applied (a clamped coordinate has both its bracketing indices land
    on the same, replicated edge row/column). Returns (II, JJ, W), each
    (4, n_pts): int32 neighbour indices and float64 weights in the order
    [(i0,j0), (i0,j1), (i1,j0), (i1,j1)]."""
    ny, nx = shape
    y, x = coords
    i0 = np.floor(y).astype(np.int64)
    j0 = np.floor(x).astype(np.int64)
    fy = y - i0
    fx = x - j0
    i0c = np.clip(i0,     0, ny - 1).astype(np.int32)
    i1c = np.clip(i0 + 1, 0, ny - 1).astype(np.int32)
    j0c = np.clip(j0,     0, nx - 1).astype(np.int32)
    j1c = np.clip(j0 + 1, 0, nx - 1).astype(np.int32)
    II = np.array([i0c, i0c, i1c, i1c])
    JJ = np.array([j0c, j1c, j0c, j1c])
    W  = np.array([(1 - fy) * (1 - fx), (1 - fy) * fx,
                    fy * (1 - fx),       fy * fx])
    return II, JJ, W


# ---------------------------------------------------------------------------
# Transect builders -- box-average analogues of the build_transect /
# build_mid_transect helpers duplicated across the source scripts
# ---------------------------------------------------------------------------
def build_box_transect(lon, lat, mask_rho, eta0, xi0, slope, depth_lim,
                        length_xi, n_pts, half_width=HALF_WIDTH_PTS, dz=DZ,
                        name=None, title=None):
    """Diagonal transect (goes in -xi direction), plus a perpendicular box
    of offset transects for box-averaging. 'coords'/'lon'/'lat'/'mask' stay
    the CENTRE line (so the x-axis / preview plotting is unchanged);
    'coords_box'/'mask_box' are the per-offset lines used by boxavg_section
    and boxavg_section_fixedz."""
    xi     = np.linspace(xi0, xi0 - length_xi, n_pts)
    eta    = np.linspace(eta0, eta0 + slope * (-length_xi), n_pts)
    coords = np.array([eta, xi])
    ne, nx = perp_unit(eta, xi)
    shape  = mask_rho.shape

    coords_box, mask_box, gather_box = [], [], []
    for s in box_offsets(half_width):
        c = np.array([eta + s * ne, xi + s * nx])
        coords_box.append(c)
        mask_box.append(map_coordinates(mask_rho.astype(float), c,
                                        order=0, mode='nearest') > 0.5)
        # Build-time invariant: the gather in interp_section clamps
        # neighbour indices to the domain, which reproduces
        # mode='nearest' exactly only because the real transect geometry
        # never actually reaches the domain edge (verified: ts eta
        # 263-479/xi 287-549, tn eta 673-841/xi 391-651, domain 1202x702).
        # Assert it here so a future ETA0/XI0/SLOPE/HALF_WIDTH_PTS change
        # that pushes the box off-domain fails loudly instead of quietly
        # producing edge-replicated (rather than out-of-bounds) samples.
        assert c[0].min() >= 0 and c[0].max() <= shape[0] - 1, \
            f'{name}: offset eta out of domain [0, {shape[0] - 1}]'
        assert c[1].min() >= 0 and c[1].max() <= shape[1] - 1, \
            f'{name}: offset xi out of domain [0, {shape[1] - 1}]'
        gather_box.append(_bilinear_gather(c, shape))
    edge_lo, edge_hi = coords_box[0], coords_box[-1]

    return dict(
        mode      = 'diag',
        name      = name,
        title     = title,
        coords    = coords,
        lon       = map_coordinates(lon, coords, order=1, mode='nearest'),
        lat       = map_coordinates(lat, coords, order=1, mode='nearest'),
        mask      = map_coordinates(mask_rho.astype(float), coords,
                                    order=0, mode='nearest') > 0.5,
        depth_lim = depth_lim,
        eta0=eta0, xi0=xi0, slope=slope,
        n_pts       = n_pts,
        coords_box  = coords_box,
        mask_box    = mask_box,
        gather_box  = gather_box,
        zgrid       = make_zgrid(depth_lim, dz),
        edge_lo_lon = map_coordinates(lon, edge_lo, order=1, mode='nearest'),
        edge_lo_lat = map_coordinates(lat, edge_lo, order=1, mode='nearest'),
        edge_hi_lon = map_coordinates(lon, edge_hi, order=1, mode='nearest'),
        edge_hi_lat = map_coordinates(lat, edge_hi, order=1, mode='nearest'),
    )


def build_box_mid_transect(lon, lat, mask_rho, eta_slice, depth_lim,
                            half_width=HALF_WIDTH_PTS, dz=DZ,
                            xlim=None, xticks=None, name='mid', title=None):
    """Geographical cross-section at a fixed eta index. Perpendicular to a
    fixed-eta row is just eta, so the box is eta_slice +/- offsets -- exact
    integer row indexing, no interpolation needed for the box itself."""
    eta_rows = np.clip(eta_slice + box_offsets(half_width),
                       0, lon.shape[0] - 1)
    lon_row = lon[eta_slice, :]
    n_pts   = lon_row.shape[0]

    return dict(
        mode      = 'mid',
        name      = name,
        title     = title,
        eta_slice = eta_slice,
        lon       = lon_row,
        depth_lim = depth_lim,
        n_pts     = n_pts,
        xlim      = xlim,
        xticks    = xticks,
        eta_rows    = eta_rows,
        zgrid       = make_zgrid(depth_lim, dz),
        edge_lo_lon = lon[eta_rows[0], :],
        edge_lo_lat = lat[eta_rows[0], :],
        edge_hi_lon = lon[eta_rows[-1], :],
        edge_hi_lat = lat[eta_rows[-1], :],
    )


# ---------------------------------------------------------------------------
# Interpolation helpers (identical logic to the interp_transect/interp_section
# duplicated in every source script, generalized to a passed-in n_pts)
#
# _ref versions below are the original per-level map_coordinates
# implementation, kept ONLY so the __main__ self-test can assert the fast
# gather-based versions reproduce them bit-for-bit (up to float summation
# order) -- not used on the hot path.
# ---------------------------------------------------------------------------
def _interp_transect_ref(field2d, coords, mask_t):
    nan_mask  = np.isnan(field2d)
    filled    = np.where(nan_mask, 0.0, field2d)
    row       = map_coordinates(filled, coords, order=1, mode='nearest')
    nan_along = map_coordinates(nan_mask.astype(float), coords,
                                order=1, mode='nearest') > 0.5
    row[nan_along | ~mask_t] = np.nan
    return row


def _interp_section_ref(field3d, coords, mask_t, n_pts):
    """Interpolate a 3D (nz, eta, xi) field along a transect. Returns (nz, n_pts)."""
    n_z = field3d.shape[0]
    out = np.full((n_z, n_pts), np.nan)
    for iz in range(n_z):
        out[iz] = _interp_transect_ref(field2d=field3d[iz], coords=coords, mask_t=mask_t)
    return out


def interp_section(field3d, gather, mask_t):
    """Interpolate a 3D (nz, eta, xi) field along a transect, using a
    precomputed (II, JJ, W) gather from _bilinear_gather (built once per
    offset in build_box_transect). Returns (nz, n_pts).

    Reproduces _interp_section_ref/_interp_transect_ref's exact NaN
    handling in a single vectorized pass over all depth levels: a NaN
    neighbour is filled with 0.0 BEFORE weighting (not simply propagated),
    matching the original 'fill then interpolate' order, and the output is
    only NaN'd where the NaN-carrying neighbours' combined weight exceeds
    0.5 or mask_t excludes the point -- NOT wherever any neighbour is NaN.
    This is not a corner case: in boxavg_section_fixedz, field3d is NaN
    below local bathymetry at every level, so this >0.5 rule (rather than a
    naive any-NaN-propagates rule) is what determines the seafloor contour
    of every *_diff_box figure. Do not change these semantics without also
    deleting the *_box_cache_*.npz caches those figures read from."""
    II, JJ, W = gather                                   # (4, n_pts) each
    v      = field3d.transpose(1, 2, 0)[II, JJ]          # (4, n_pts, nz)
    nanv   = np.isnan(v)
    filled = np.where(nanv, 0.0, v)
    Wc     = W[:, :, None]                               # (4, n_pts, 1)
    out    = (Wc * filled).sum(axis=0)                   # (n_pts, nz)
    bad    = ((Wc * nanv).sum(axis=0) > 0.5) | ~mask_t[:, None]
    out[bad] = np.nan
    return out.T                                         # (nz, n_pts)


def interp_to_z(fld, zr, zgrid):
    """(nz, npts) field on per-column zr -> (len(zgrid), npts) on the common
    zgrid. Per column: np.interp(zgrid, zr[:,j], fld[:,j]), so cells below
    the local seafloor / above the local surface fall outside [zr.min(),
    zr.max()] and come back NaN (left=right=nan) rather than being
    (wrongly) extrapolated. Columns with any NaN in fld or zr (land, from
    the ~mask_t masking in interp_transect) are skipped entirely -- zr and
    fld share the same mask_t, so this is an all-or-nothing land/water call
    per column, not a partial-column drop."""
    nz, npts = fld.shape
    out = np.full((len(zgrid), npts), np.nan)
    for j in range(npts):
        zrj, fldj = zr[:, j], fld[:, j]
        if np.isnan(zrj).any() or np.isnan(fldj).any():
            continue
        out[:, j] = np.interp(zgrid, zrj, fldj, left=np.nan, right=np.nan)
    return out


# ---------------------------------------------------------------------------
# Box averages
# ---------------------------------------------------------------------------
def boxavg_section(field3d, zr3d, tr):
    """Native s_rho/s_w field3d + its per-scenario zr3d (nz, eta, xi) ->
    box-averaged section on tr['zgrid'] (len(zgrid), tr['n_pts']). For each
    offset in the box: extract the section, interpolate to zgrid, then
    accumulate with an explicit sum/count (not np.nanmean) so all-NaN
    columns come back NaN without RuntimeWarning spam, and so MIN_FRAC can
    be applied. Cells below the local seafloor drop out of the count at
    that depth, so the average only uses water that actually exists there."""
    zgrid = tr['zgrid']
    n_pts = tr['n_pts']
    sums   = np.zeros((len(zgrid), n_pts))
    counts = np.zeros((len(zgrid), n_pts))

    if tr['mode'] == 'diag':
        offsets = list(zip(tr['gather_box'], tr['mask_box']))
        n_off = len(offsets)
        for gather, mask_t in offsets:
            fld_s = interp_section(field3d, gather, mask_t)
            zr_s  = interp_section(zr3d,    gather, mask_t)
            fld_z = interp_to_z(fld_s, zr_s, zgrid)
            valid = ~np.isnan(fld_z)
            sums[valid]   += fld_z[valid]
            counts[valid] += 1
    else:   # 'mid'
        n_off = len(tr['eta_rows'])
        for row in tr['eta_rows']:
            fld_s = field3d[:, row, :]
            zr_s  = zr3d[:, row, :]
            fld_z = interp_to_z(fld_s, zr_s, zgrid)
            valid = ~np.isnan(fld_z)
            sums[valid]   += fld_z[valid]
            counts[valid] += 1

    with np.errstate(invalid='ignore'):
        out = sums / counts
    if MIN_FRAC > 0:
        out[counts < MIN_FRAC * n_off] = np.nan
    return out


def boxavg_section_fixedz(field3d, tr, min_frac=None):
    """Fixed-depth-grid variant for the zsliced *_diff.py scripts. field3d
    is (n_depth, eta, xi) on the SAME depth levels at every time step, so no
    vertical interpolation is needed -- extract each offset's section and
    average across offsets with the same sum/count logic. field3d is
    already NaN on land and below local bathymetry (the zslicer's fill
    value has already been converted to NaN upstream), so shallower offsets
    simply drop out of the count at the depths they don't reach.

    min_frac overrides the module-level MIN_FRAC for this call (None uses
    MIN_FRAC, i.e. no change for existing callers). Right at a transect
    point's deepest reachable level, support can collapse from all n_off
    offsets to just one or two -- and that thin, unrepresentative remainder
    can differ sharply from the well-supported mean one level up, producing
    a spurious spike (seen concretely in drho/dz: 21/21 offsets agreeing on
    sigma-t 25.22 at -28 m, then a single surviving offset giving 13.39 at
    -29 m). Passing e.g. min_frac=0.5 NaNs out those thin levels instead of
    averaging them in.

    Returns (n_depth, tr['n_pts'])."""
    n_pts   = tr['n_pts']
    n_depth = field3d.shape[0]
    sums   = np.zeros((n_depth, n_pts))
    counts = np.zeros((n_depth, n_pts))

    if tr['mode'] == 'diag':
        offsets = list(zip(tr['gather_box'], tr['mask_box']))
        n_off = len(offsets)
        for gather, mask_t in offsets:
            fld_s = interp_section(field3d, gather, mask_t)
            valid = ~np.isnan(fld_s)
            sums[valid]   += fld_s[valid]
            counts[valid] += 1
    else:   # 'mid'
        n_off = len(tr['eta_rows'])
        for row in tr['eta_rows']:
            fld_s = field3d[:, row, :]
            valid = ~np.isnan(fld_s)
            sums[valid]   += fld_s[valid]
            counts[valid] += 1

    with np.errstate(invalid='ignore'):
        out = sums / counts
    frac = MIN_FRAC if min_frac is None else min_frac
    if frac > 0:
        out[counts < frac * n_off] = np.nan
    return out


def box_edges(tr):
    """((lon, lat), (lon, lat)) of the two box edges -- for drawing the
    averaged footprint on map panels or the interactive transect preview.
    Both 'diag' and 'mid' transects carry the same edge_lo_*/edge_hi_* keys."""
    return ((tr['edge_lo_lon'], tr['edge_lo_lat']),
            (tr['edge_hi_lon'], tr['edge_hi_lat']))


# ---------------------------------------------------------------------------
# Self-test -- runs only when this module is executed directly, not on
# `import cs_boxavg as cb` from the _box.py scripts. Needs only the grid
# file (no model output), so it can be run standalone as a sanity check.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from netCDF4 import Dataset

    print(f'GRD     = {GRD}')
    print(f'figpath = {figpath("snapshots", "example.png")}')
    assert os.path.basename(PLOT_DIR) == 'plot', \
        'PLOT_DIR did not resolve to .../plot -- path anchoring is broken'
    assert os.path.exists(GRD), f'grid file not found at {GRD}'
    assert figpath('x.png') == os.path.join(HERE, 'figs', 'x.png'), \
        'figpath() did not resolve under boxavg/figs -- path anchoring is broken'

    grdnc    = Dataset(GRD, 'r')
    lat      = np.array(grdnc['lat_rho'][:])
    lon      = np.array(grdnc['lon_rho'][:]) - 360
    mask_rho = np.array(grdnc['mask_rho'][:])
    eta_rho, xi_rho = lat.shape

    # --- perp_unit direction check -----------------------------------------
    ETA0, XI0, SLOPE, LENGTH_XI, N_PTS = 271, 543, -0.8, 250, 300
    ETA1, XI1, SLOPE1 = 832, 646, 0.6

    def along(eta0, xi0, slope):
        xi  = np.linspace(xi0, xi0 - LENGTH_XI, N_PTS)
        eta = np.linspace(eta0, eta0 + slope * (-LENGTH_XI), N_PTS)
        return eta, xi

    for label, (eta0, xi0, slope), expect in [
        ('ts', (ETA0, XI0, SLOPE),  (-0.781, -0.625)),
        ('tn', (ETA1, XI1, SLOPE1), (-0.857,  0.514)),
    ]:
        eta, xi = along(eta0, xi0, slope)
        ne, nx = perp_unit(eta, xi)
        print(f'{label}: perp_unit = ({ne:.3f}, {nx:.3f})  expect {expect}')
        assert abs(ne - expect[0]) < 1e-2 and abs(nx - expect[1]) < 1e-2

        along_vec = np.array([eta[-1] - eta[0], xi[-1] - xi[0]])
        along_vec = along_vec / np.hypot(*along_vec)
        dot = ne * along_vec[0] + nx * along_vec[1]
        assert abs(dot) < 1e-9, f'{label}: perp not orthogonal, dot={dot}'

    # --- box transects stay inside the domain -------------------------------
    # (build_box_transect itself now asserts a stricter [0, N-1] bound on
    # every offset -- see the assert inside the function -- so getting here
    # at all already proves that invariant held; this loop is a second,
    # independent check plus the [-1, N] tolerance the self-test has always used.)
    rng = np.random.default_rng(42)
    for label, (eta0, xi0, slope, depth_lim) in [
        ('ts', (ETA0, XI0, SLOPE,  -125)),
        ('tn', (ETA1, XI1, SLOPE1, -90)),
    ]:
        tr = build_box_transect(lon, lat, mask_rho, eta0, xi0, slope, depth_lim,
                                LENGTH_XI, N_PTS, name=label)
        for c in tr['coords_box']:
            assert c[0].min() >= -1 and c[0].max() <= eta_rho, \
                f'{label}: offset eta out of bounds'
            assert c[1].min() >= -1 and c[1].max() <= xi_rho, \
                f'{label}: offset xi out of bounds'
        print(f'{label}: {len(tr["coords_box"])} offsets, all within domain '
              f'(eta_rho={eta_rho}, xi_rho={xi_rho})')

        # --- gather-based interp_section reproduces the reference
        # per-level map_coordinates implementation, including the 0-fill /
        # >0.5-weight NaN rule (cs_boxavg.py module docstring). Use a
        # synthetic field carrying BOTH land NaNs (from the real mask) and
        # a seafloor-like interior NaN band, since a naive
        # any-neighbour-NaN-propagates gather would only diverge from the
        # reference in exactly that partial-weight case.
        nz_test = 12
        field = rng.standard_normal((nz_test,) + mask_rho.shape)
        field[:, mask_rho == 0] = np.nan
        e0, e1 = int(tr['coords_box'][0][0].min()), int(tr['coords_box'][-1][0].max())
        x0, x1 = int(tr['coords_box'][0][1].min()), int(tr['coords_box'][-1][1].max())
        field[:nz_test // 2, e0:e0 + 20, x0:x0 + 20] = np.nan
        for gather, mask_t, coords in zip(tr['gather_box'], tr['mask_box'],
                                          tr['coords_box']):
            ref = _interp_section_ref(field, coords, mask_t, N_PTS)
            new = interp_section(field, gather, mask_t)
            assert np.array_equal(np.isnan(ref), np.isnan(new)), \
                f'{label}: interp_section NaN pattern differs from reference'
            g = ~np.isnan(ref)
            assert np.allclose(ref[g], new[g], rtol=0, atol=1e-9), \
                f'{label}: interp_section values differ from reference'
        print(f'{label}: interp_section (gather) matches reference on '
              f'{len(tr["gather_box"])} offsets, {nz_test} levels, '
              f'land + seafloor NaN structure')

    tr_mid = build_box_mid_transect(lon, lat, mask_rho, 477, -300)
    assert tr_mid['eta_rows'].min() >= 0 and tr_mid['eta_rows'].max() < eta_rho
    print(f'mid: eta_rows {tr_mid["eta_rows"].min()}..{tr_mid["eta_rows"].max()}, '
          f'within domain')

    # --- interp_to_z round-trip ---------------------------------------------
    zgrid = make_zgrid(-100, dz=2.0)
    npts  = 5
    zr_col = np.linspace(-120, -1, 60)          # increasing, like Cs_r*h
    zr = np.tile(zr_col[:, None], (1, npts))
    fld = zr.copy()                              # feed zr itself as the field
    out = interp_to_z(fld, zr, zgrid)
    in_range = (zgrid >= zr_col.min()) & (zgrid <= zr_col.max())
    assert np.allclose(out[in_range, :], zgrid[in_range, None], atol=1e-6)
    assert np.all(np.isnan(out[~in_range, :]))
    print('interp_to_z round-trip OK '
          f'({in_range.sum()}/{len(zgrid)} zgrid levels in range)')

    print('All cs_boxavg self-tests passed.')
