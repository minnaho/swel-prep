function fld = get_data_slice(GridInfo, varname, irec, method)
% Generic function to read a slice of NetCDF data and interpolate it
%
% GridInfo: Struct containing source/target coordinates and file info
% varname:  String name of variable in NetCDF
% irec:     Time record index to read
% method:   Interpolation method ('linear', 'makima', 'none')
%           If 'none', returns raw data (inpainted) without interpolation.

    datname = GridInfo.datname;
    
    % Input Source Grid
    src_x = GridInfo.src_x;
    src_y = GridInfo.src_y;
    
    % Target Grid
    tgt_x = GridInfo.tgt_x;
    tgt_y = GridInfo.tgt_y;
    
    i0 = GridInfo.i0; j0 = GridInfo.j0;
    fnx = GridInfo.fnx; fny = GridInfo.fny;
    mask = GridInfo.mask;

    % --- Read Data ---
    try
        % Standard read: Start=[i0, j0, irec], Count=[fnx, fny, 1]
        % Matlabs ncread usually preserves (xi, eta) ordering.
        start_vec = [i0, j0, irec];
        count_vec = [fnx, fny, 1];
        frc = ncread(datname, varname, start_vec, count_vec);
    catch
        % Fallback for dimension ordering issues
        warning(['Standard read failed for ' varname '. Reading all and slicing...']);
        frc_all = ncread(datname, varname);
        % Heuristic: Assume time is last dim
        if ndims(frc_all) == 3
            frc = frc_all(:,:,irec);
        else
            error('Unknown dimension structure');
        end
    end
    
    frc = double(frc);
    
    % --- Masking & Inpainting ---
    % If a mask is provided (1=water, 0=land), set land to NaN so they fill
    % properly during inpainting, rather than interpolating zeros (land) into water.
    if ~isempty(mask)
        if size(mask) == size(frc)
            frc(mask < 0.5) = NaN;
        end
    end
    
    % Inpaint NaNs (Land) to prevent interpolation errors
    if any(isnan(frc(:)))
         % Use method 2 (del^2) for smooth filling of land/missing chunks
         frc = inpaint_nans(frc, 2); 
    end

    % --- Interpolation ---
    
    % If method is 'none', we return the raw (inpainted) data.
    % This is used in the driver when we need to do math (like vector rotation)
    % BEFORE interpolating.
    if strcmp(method, 'none')
        fld = frc;
        return;
    end

    % --- Robust Interpolation Strategy ---
    try
        % Attempt standard interp2 (Fastest, but requires Orthogonal/Plaid grid)
        % Note: For 2D matrix inputs X, Y, V, they must all be the same size.
        % frc matches src_x/src_y dimensions directly (xi, eta).
        fld = interp2(src_x, src_y, frc, tgt_x, tgt_y, method, NaN);
        
    catch 
        % Handle Curvilinear Grids (Rotated/Warped ROMS grids)
        % interp2 fails if the mesh isn't strictly monotonic/orthogonal.
        % We catch ALL errors here to ensure fallback happens.
       
        % Map unsupported grid methods to Scattered equivalents
        % 'makima'/'spline' require grids; 'natural' is the smooth equivalent for scattered
        if any(strcmp(method, {'makima', 'spline', 'cubic'}))
            scat_method = 'natural';
        else
            scat_method = method;
        end
        
        % Use scatteredInterpolant (Works for any 2D shape)
        % Ensure arrays are column vectors and values correspond index-wise
        % src_x/y and frc are all (xi, eta), so we flatten directly.
        F = scatteredInterpolant(src_x(:), src_y(:), frc(:), scat_method, 'nearest');
        fld = F(tgt_x, tgt_y);
    end
    
    % Fill edges of result if any NaNs remain (e.g., slight boundary mismatches)
    if any(isnan(fld(:)))
         fld = inpaint_nans(fld, 0);
    end
    
    return
