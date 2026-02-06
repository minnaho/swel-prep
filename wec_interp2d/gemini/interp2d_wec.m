%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  ROMS Wave Forcing Interpolator (Coarse -> Fine)
%
%  Purpose: Interpolate wave forcing from a coarse ROMS grid to a 
%           finer ROMS grid with smoothing.
%
%  UPDATE: 
%  - Reverted to 'interp2' as requested.
%  - Projection limits encompass SOURCE grid to prevent NaNs (Fixes previous error).
%  - Preserves True North fidelity for vectors.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; conf.quiet = true;

% --- CONFIGURATION --------------------------------------------------

% 1. Files
frc_dir      = '../';       
frc_file     = 'smode_frc_ww3_20190415.nc'; 

% 2. Grids
% NOTE: The forcing file usually doesn't contain lat/lon. 
% You must point to the grid file used to generate the source forcing.
grdname_src  = '../smode200_grd.nc'; % 200m Grid (Source)
grdname_tgt  = '../mc60_grd.nc';     % Target Grid (Fine)

root_name    = 'mc60';               % Output prefix

% 3. Interpolation Method
% 'makima' works well with interp2 for smoothing.
interp_method = 'makima'; 

% 4. Variable Mapping
VarMap.time_name = 'wwv_time';

% Scalar Variables
VarMap.Awave = 'Awave'; 
VarMap.Dwave = 'Dwave'; 
VarMap.Pwave = 'Pwave'; 
VarMap.eb    = 'eb';    
VarMap.ed    = 'ed';    
VarMap.lmw   = 'lmw';   
VarMap.qb    = 'qb';    
VarMap.sup   = 'sup';   

% Vector Variables (STAY TRUE NORTH)
VecMap(1).tgt = {'uorb', 'vorb'};
VecMap(1).src = {'uorb', 'vorb'};

VecMap(2).tgt = {'ust0', 'vst0'};
VecMap(2).src = {'ust0', 'vst0'};

VecMap(3).tgt = {'ust2d', 'vst2d'};
VecMap(3).src = {'ust2d', 'vst2d'};

% --------------------------------------------------------------------

% --- SETUP GRIDS ---

% 1. Load Target Grid
disp('Reading Target Grid (Fine)...');
lon_tgt = ncread(grdname_tgt, 'lon_rho');
lat_tgt = ncread(grdname_tgt, 'lat_rho');
[nx_tgt, ny_tgt] = size(lon_tgt);
lon_tgt = mod(lon_tgt, 360);

% 2. Load Source Grid
disp('Reading Source Grid (Coarse)...');
try
    lon_src = ncread(grdname_src, 'lon_rho');
    lat_src = ncread(grdname_src, 'lat_rho');
    try
        mask_src = ncread(grdname_src, 'mask_rho');
    catch
        mask_src = []; 
    end
catch
    error('Could not read grid coordinates from grdname_src. Ensure path is correct.');
end
lon_src = mod(lon_src, 360);

% 3. Project to Cartesian Plane (Lambert)
% NOTE: We use m_map to get Cartesian coordinates (meters) to avoid
% issues with lat/lon scaling differences.
disp('Setting up Projection...');
addpath('/data/project9/minnaho/ucla-tools/frc/m_map'); 

% CRITICAL FIX: Set limits based on SOURCE grid + Buffer
% This prevents NaNs/Infs that crash interp2 when source data is clipped.
min_lon = min(lon_src(:)) - 1.0;
max_lon = max(lon_src(:)) + 1.0;
min_lat = min(lat_src(:)) - 1.0;
max_lat = max(lat_src(:)) + 1.0;

mid_lat = mean(lat_src(:));
mid_lon = mean(lon_src(:));

m_proj('lambert','lon',[min_lon max_lon], ...
      'lat',[min_lat max_lat], ...
      'clo', mid_lon, 'par', [mid_lat-1 mid_lat+1], 'ell','sphere');

mult = 1000; 
[xf, yf] = m_ll2xy(lon_src, lat_src);
[xg, yg] = m_ll2xy(lon_tgt, lat_tgt);
xf = xf * mult; yf = yf * mult;
xg = xg * mult; yg = yg * mult;

% Store in GridInfo for the worker function
GridInfo.datname = [frc_dir frc_file];
GridInfo.src_x = xf;
GridInfo.src_y = yf;
GridInfo.tgt_x = xg;
GridInfo.tgt_y = yg;
GridInfo.i0 = 1; GridInfo.j0 = 1;
[GridInfo.fnx, GridInfo.fny] = size(lon_src);
GridInfo.mask = mask_src; 

% --- PROCESS FILE ---------------------------------------------------

datname = [frc_dir frc_file];
disp(['Processing: ' frc_file]);

time_var = VarMap.time_name;
raw_time = ncread(datname, time_var);
nrecord = length(raw_time);

frcname = [root_name '_wec.20190415.nc'];
create_frc_wave(grdname_tgt, frcname);

disp(['Interpolating ' num2str(nrecord) ' records...']);

for irec = 1:nrecord
    disp(['  Record: ' num2str(irec)]);
    
    % 1. Time
    ncwrite(frcname, 'wwv_time', raw_time(irec), irec);
    
    % 2. Scalar Variables
    scalar_fields = {'Awave', 'Pwave', 'eb', 'ed', 'lmw', 'qb', 'sup'};
    for k = 1:length(scalar_fields)
        fld_name = scalar_fields{k};
        src_name = VarMap.(fld_name);
        
        dat = get_data_slice(GridInfo, src_name, irec, interp_method);
        ncwrite(frcname, fld_name, dat, [1 1 irec]);
    end
    
    % 3. Direction (Special Handling)
    D_src = get_data_slice(GridInfo, VarMap.Dwave, irec, 'none'); 
    % Fill physical NaNs (from land mask) before math
    if any(isnan(D_src(:))), D_src = inpaint_nans(D_src, 2); end
    
    D_rad = D_src * pi / 180;
    U_dir = cos(D_rad);
    V_dir = sin(D_rad);
    
    % Use interp2 directly for unit vectors
    % Note: xf, yf are passed as matrices. interp2 expects V to match dimensions.
    % get_data_slice handles the transpose of V internally. Here we manually transpose.
    U_interp = interp2(xf, yf, U_dir', xg, yg, interp_method, NaN);
    V_interp = interp2(xf, yf, V_dir', xg, yg, interp_method, NaN);
    
    D_tgt = atan2(V_interp, U_interp) * 180 / pi;
    D_tgt = mod(D_tgt, 360);
    
    if any(isnan(D_tgt(:))), D_tgt = inpaint_nans(D_tgt, 0); end
    ncwrite(frcname, 'Dwave', D_tgt, [1 1 irec]);
    
    % 4. Vector Variables
    for v = 1:length(VecMap)
        u_name = VecMap(v).src{1};
        v_name = VecMap(v).src{2};
        
        u_src = get_data_slice(GridInfo, u_name, irec, 'none');
        v_src = get_data_slice(GridInfo, v_name, irec, 'none');
        
        if any(isnan(u_src(:))), u_src = inpaint_nans(u_src, 2); end
        if any(isnan(v_src(:))), v_src = inpaint_nans(v_src, 2); end
        
        % Direct Interpolation using interp2
        u_tgt = interp2(xf, yf, u_src', xg, yg, interp_method, NaN);
        v_tgt = interp2(xf, yf, v_src', xg, yg, interp_method, NaN);
        
        if any(isnan(u_tgt(:))), u_tgt = inpaint_nans(u_tgt, 0); end
        if any(isnan(v_tgt(:))), v_tgt = inpaint_nans(v_tgt, 0); end

        ncwrite(frcname, VecMap(v).tgt{1}, u_tgt, [1 1 irec]);
        ncwrite(frcname, VecMap(v).tgt{2}, v_tgt, [1 1 irec]);
    end

end

disp('Done.');
