addpath('/data/project3/minnaho/project9copy/ucla-tools/ini');

gname = '../../mc60_grd.nc';
 %path  = '/data/project3/minnaho/swel/tides/mc60/wec/his/';
 %path  = '/data/project3/minnaho/swel/tides/mc60/nowec/output/his/';
 path  = '/data/project3/minnaho/swel/notides/mc60/nowec/output/his/';
 list = dir([path 'mc60_his.*.nc']); % Added .nc to be specific
 nfiles = length(list);

 h = ncread(gname,'h');
 f = ncread(gname,'f');
 pm = ncread(gname,'pm');
 pn = ncread(gname,'pn');
 dx = 1./pm;
 dy = 1./pn;

 [nx, ny] = size(dx);
 fname_first = [path list(1).name];
 % Determine nz by checking the dimension of a 3D variable
 info = ncinfo(fname_first, 'temp');
 nz = info.Size(3);

 for i = 1:nfiles
    if contains(list(i).name, '_pv')
        continue;
    end
    % Input file
    fname_in = [path list(i).name];
    % Output file name (e.g., mc60_his.0000_pv.nc)
    fname_out = strrep(fname_in, '.nc', '_pv.nc');
    
    fprintf('Processing %s -> %s\n', list(i).name, fname_out);

    tim = ncread(fname_in, 'ocean_time');
    nfr = length(tim);

    % 1. Skip this file entirely if the output file already exists
    if exist(fname_out, 'file')
        fprintf('Skipping %s: output %s already exists.\n', list(i).name, fname_out);
        continue;
    end
    % Define Dimensions
    nccreate(fname_out, 'pv', ...
             'Dimensions', {'xi_rho', nx, 'eta_rho', ny, 's_rho', nz, 'ocean_time', Inf}, ...
             'Datatype', 'single');
    
    % Define ocean_time in the new file so the axis is preserved
    nccreate(fname_out, 'ocean_time', 'Dimensions', {'ocean_time', Inf}, 'Datatype', 'double');

    % Write Attributes
    ncwriteatt(fname_out, 'pv', 'long_name', 'Potential vorticity');
    ncwriteatt(fname_out, 'pv', 'units', 'm/s3');
    ncwriteatt(fname_out, 'ocean_time', 'units', 'seconds since 1900-01-01 00:00:00');

    % 2. Loop through time frames
    for ifr = 1:nfr
        fprintf('  Time frame: %d/%d\n', ifr, nfr);
        
        % Read data from input file
        u = ncread(fname_in, 'u', [1 1 1 ifr], [inf inf inf 1]);
        v = ncread(fname_in, 'v', [1 1 1 ifr], [inf inf inf 1]);
        temp = ncread(fname_in, 'temp', [1 1 1 ifr], [inf inf inf 1]);
        salt = ncread(fname_in, 'salt', [1 1 1 ifr], [inf inf inf 1]);
        zeta = ncread(fname_in, 'zeta', [1 1 ifr], [inf inf 1]);
        
        % Calculate vertical coordinates
        zr = zlevs(h, zeta, fname_in, 'r');
        zw = zlevs(h, zeta, fname_in, 'w');
        dz = zw(:,:,2:end) - zw(:,:,1:end-1);

        % Calculate PV
        pv = calc_pv(u, v, temp, salt, zr, dz, dx, dy, f);

        % 3. Write to the NEW file
        ncwrite(fname_out, 'pv', pv, [1 1 1 ifr]);
        ncwrite(fname_out, 'ocean_time', tim(ifr), ifr);
    end
 end
