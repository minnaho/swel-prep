function create_frc_wave(gridfile, frcfile)
%
%  Create ROMS Wave Forcing file (SMODE/WW3 Format)
%
%  Inputs:
%    gridfile   - Path to Target ROMS grid file
%    frcfile    - Path to output forcing file
%

% --- Get Dimensions ---
try
    h = ncread(gridfile, 'h');
catch
    error('Grid variable "h" not found in grid file.');
end
[nx, ny] = size(h);

% --- Create File ---
if exist(frcfile, 'file')
    delete(frcfile); 
end

% --- Dimensions ---
nccreate(frcfile, 'wwv_time', 'Dimensions', {'wwv_time', 0}, 'datatype', 'double');
ncwriteatt(frcfile, 'wwv_time', 'long_name', 'surface gravity wave time');
ncwriteatt(frcfile, 'wwv_time', 'units', 'days'); 
% Note: Using "days" generally, specific epoch depends on ROMS clock

% --- Variables ---

% 1. Wave Amplitude
nccreate(frcfile, 'Awave', 'Dimensions', {'xi_rho', nx, 'eta_rho', ny, 'wwv_time', 0}, 'datatype', 'single');
ncwriteatt(frcfile, 'Awave', 'long_name', 'mean wind wave amplitude (Hsig/2/sqrt(2))');
ncwriteatt(frcfile, 'Awave', 'units', 'meter');

% 2. Wave Direction
nccreate(frcfile, 'Dwave', 'Dimensions', {'xi_rho', nx, 'eta_rho', ny, 'wwv_time', 0}, 'datatype', 'single');
ncwriteatt(frcfile, 'Dwave', 'long_name', 'clockwise mean wind wave direction from true north');
ncwriteatt(frcfile, 'Dwave', 'units', 'degree');

% 3. Period
nccreate(frcfile, 'Pwave', 'Dimensions', {'xi_rho', nx, 'eta_rho', ny, 'wwv_time', 0}, 'datatype', 'single');
ncwriteatt(frcfile, 'Pwave', 'long_name', 'Peak wave period (tp)');
ncwriteatt(frcfile, 'Pwave', 'units', 'seconds');

% 4. Dissipation (Breaking)
nccreate(frcfile, 'eb', 'Dimensions', {'xi_rho', nx, 'eta_rho', ny, 'wwv_time', 0}, 'datatype', 'single');
ncwriteatt(frcfile, 'eb', 'long_name', 'breaking dissipation');
ncwriteatt(frcfile, 'eb', 'units', 'm3/s3');

% 5. Dissipation (Friction)
nccreate(frcfile, 'ed', 'Dimensions', {'xi_rho', nx, 'eta_rho', ny, 'wwv_time', 0}, 'datatype', 'single');
ncwriteatt(frcfile, 'ed', 'long_name', 'bed frictional energy dissipation');
ncwriteatt(frcfile, 'ed', 'units', 'm3/s3');

% 6. Wavelength
nccreate(frcfile, 'lmw', 'Dimensions', {'xi_rho', nx, 'eta_rho', ny, 'wwv_time', 0}, 'datatype', 'single');
ncwriteatt(frcfile, 'lmw', 'long_name', 'mean wavelength');
ncwriteatt(frcfile, 'lmw', 'units', 'm');

% 7. Breaking Fraction
nccreate(frcfile, 'qb', 'Dimensions', {'xi_rho', nx, 'eta_rho', ny, 'wwv_time', 0}, 'datatype', 'single');
ncwriteatt(frcfile, 'qb', 'long_name', 'fraction of breaking waves');
ncwriteatt(frcfile, 'qb', 'units', 'nondimensional');

% 8. Set Down
nccreate(frcfile, 'sup', 'Dimensions', {'xi_rho', nx, 'eta_rho', ny, 'wwv_time', 0}, 'datatype', 'single');
ncwriteatt(frcfile, 'sup', 'long_name', 'set down');
ncwriteatt(frcfile, 'sup', 'units', 'm');

% --- VECTORS ---

% 9. Orbital Velocity
nccreate(frcfile, 'uorb', 'Dimensions', {'xi_rho', nx, 'eta_rho', ny, 'wwv_time', 0}, 'datatype', 'single');
ncwriteatt(frcfile, 'uorb', 'long_name', 'Eastward bottom orbital velocity'); 
ncwriteatt(frcfile, 'uorb', 'units', 'm/s');

nccreate(frcfile, 'vorb', 'Dimensions', {'xi_rho', nx, 'eta_rho', ny, 'wwv_time', 0}, 'datatype', 'single');
ncwriteatt(frcfile, 'vorb', 'long_name', 'Northward bottom orbital velocity');
ncwriteatt(frcfile, 'vorb', 'units', 'm/s');

% 10. Surface Stokes
nccreate(frcfile, 'ust0', 'Dimensions', {'xi_rho', nx, 'eta_rho', ny, 'wwv_time', 0}, 'datatype', 'single');
ncwriteatt(frcfile, 'ust0', 'long_name', 'Eastward surface Stokes drift');
ncwriteatt(frcfile, 'ust0', 'units', 'm/s');

nccreate(frcfile, 'vst0', 'Dimensions', {'xi_rho', nx, 'eta_rho', ny, 'wwv_time', 0}, 'datatype', 'single');
ncwriteatt(frcfile, 'vst0', 'long_name', 'Northward surface Stokes drift');
ncwriteatt(frcfile, 'vst0', 'units', 'm/s');

% 11. Depth Avg Stokes
nccreate(frcfile, 'ust2d', 'Dimensions', {'xi_rho', nx, 'eta_rho', ny, 'wwv_time', 0}, 'datatype', 'single');
ncwriteatt(frcfile, 'ust2d', 'long_name', 'Eastward-Depth averaged Stokes drift');
ncwriteatt(frcfile, 'ust2d', 'units', 'm/s');

nccreate(frcfile, 'vst2d', 'Dimensions', {'xi_rho', nx, 'eta_rho', ny, 'wwv_time', 0}, 'datatype', 'single');
ncwriteatt(frcfile, 'vst2d', 'long_name', 'Northward-Depth averaged Stokes drift');
ncwriteatt(frcfile, 'vst2d', 'units', 'm/s');

% --- Global Attributes ---
ncwriteatt(frcfile, '/', 'Title', 'Interpolated Wave Forcing (SMODE)');
ncwriteatt(frcfile, '/', 'Date', datestr(now));
ncwriteatt(frcfile, '/', 'gridfile', gridfile);

return
