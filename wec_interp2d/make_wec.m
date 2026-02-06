%---------------------------------------------------------------------------------------
%
%  make_wec
%
%  Downscale WEC forcing from parent to child

%  Minna Ho 2025 at UCLA
%  Based off code from
%  Jeroen Molemaker and Evan Mason in 2007-2009 at UCLA
%
%---------------------------------------------------------------------------------------
clear all
close all
disp(' ')
%---------------------------------------------------------------------------------------
%  USER-DEFINED VARIABLES & OPTIONS START HERE
%---------------------------------------------------------------------------------------
%
%   Parent...
%
    parscd.file = '/data/project9/minnaho/swel/smode_frc_ww3_20190415.nc';
    pargrd = '/data/project9/minnaho/swel/smode200_grd.nc'
%
%   child
%
    romsdir    = '/data/project9/minnaho/swel/';
    chdgrd    = [romsdir 'mc60_grd.nc'];
    chdini    = [romsdir 'wec_interp2d/mc60_interp_wec.20190415.nc'];
%    
%---------------------------------------------------------------------------------------
% USER-DEFINED VARIABLES & OPTIONS END HERE
%---------------------------------------------------------------------------------------
%

    if ~exist(chdini)
      disp(['Creating initial file: ' chdini]);
      create_wec(chdini,chdgrd,parscd.file)
    end

    disp('begin interpolating and writing to file')
    interp_wec(pargrd, parscd.file, chdgrd, chdini)
    
