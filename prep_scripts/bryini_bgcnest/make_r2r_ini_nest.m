%---------------------------------------------------------------------------------------
%
%  make_s2r
%
%  Generate boundary perimeter file from WOA and SSH  data.
%
%  Note that when run this script it tests for the presence of a .mat file
%  which contains various interpolation coefficients related to your child
%  and parent grids.  If the .mat file is not there it will calculate the coefficients
%
%
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
%---------------------------------------------------------------------------------------
%  1.  GENERAL
%---------------------------------------------------------------------------------------
%
%   Parent...
%
     parscd.file    = '/data/project3/minnaho/project9copy/swel/notides/smode_rst.20190415110120.nc' ; 
     pargrd = '/data/project3/minnaho/project9copy/swel/smode200_grd.nc' ;
     parscd.N       = 100 ;
     parscd.theta_s = 6.0;
     parscd.theta_b = 6.0;
     parscd.hc      = 250 ;
     parscd.tind    = 1;            % frame number in parent file
     parscd.scoord = 'new2012';    % child 'new' or 'old' type scoord
    
%%%%% child
    romsdir    = '/data/project3/minnaho/project9copy/swel/notides/';
    chdgrd    = [romsdir 'mc60/mc60_grd.nc'];
    chdini    = [romsdir 'mc60/mc60_ini.20190415110120.nc'];
    chdscd.N      = 100;
    chdscd.theta_s = 6.0;
    chdscd.theta_b = 6.0;
    chdscd.hc     = 250.0;
    chdscd.scoord = 'new2012';    % child 'new' or 'old' type scoord
 
%% ADD THE DO_SLOW
    do_slow = 1 ; % write  u_slow, v_slow, p_slow from restart file.

    makebgc=1 ;    

    BGC_INI.data_roms      = 1 ; 
    BGC_INI.tracer_roms      = [1 2 3 4 5 6 7 8 9 10 11 12 ...
                                13 14 15 16 17 18 19 20 21 ...
                                22 23 24 25 26 27 28 29] ;  
    BGC_INI.file_bgc{1} = parscd.file ;     
    BGC_INI.bgc_tracer = {'NO3','PO4','SiO3','Fe','O2', 'SPC','SPCHL','SPFE','SPCACO3', ...
                  'DIATC','DIATCHL','DIATFE','DIATSI','DIAZC','DIAZCHL','DIAZFE', ...
                  'ZOOC','DOC','NH4','DIC','Alk', ...
                  'DON','DONR','DOP','DOPR','DOFE','NO2','N2','N2O'};
    BGC_INI.bgc_frctype = {'i','i','i','i','i','i','i','i','i','i','i','i', ...
                           'i','i','i','i','i','i','i','i','i', ...
                           'i','i','i','i','i','i','i','i'} ;
    BGC_INI.bgc_frcini = {BGC_INI.file_bgc{1},BGC_INI.file_bgc{1},BGC_INI.file_bgc{1},BGC_INI.file_bgc{1}, ...
                          BGC_INI.file_bgc{1},BGC_INI.file_bgc{1},BGC_INI.file_bgc{1},BGC_INI.file_bgc{1}, ...
                          BGC_INI.file_bgc{1},BGC_INI.file_bgc{1},BGC_INI.file_bgc{1},BGC_INI.file_bgc{1}, ...
                          BGC_INI.file_bgc{1},BGC_INI.file_bgc{1},BGC_INI.file_bgc{1},BGC_INI.file_bgc{1}, ...
                          BGC_INI.file_bgc{1},BGC_INI.file_bgc{1},BGC_INI.file_bgc{1},BGC_INI.file_bgc{1}, ...
                          BGC_INI.file_bgc{1}, ...
                          BGC_INI.file_bgc{1},BGC_INI.file_bgc{1},BGC_INI.file_bgc{1},BGC_INI.file_bgc{1}, ...
                          BGC_INI.file_bgc{1},BGC_INI.file_bgc{1},BGC_INI.file_bgc{1},BGC_INI.file_bgc{1}} ;          

    bgc_tracer_list ;

    sed_tracers=1 ; 
    SEDBGC_INI.data_roms      = 1 ;
    SEDBGC_INI.tracer_roms    = [1 2 3] ;
    SEDBGC_INI.file_bgc{1} = parscd.file ;
    SEDBGC_INI.bgc_tracer = {'Sed_POC','Sed_CaCO3','Sed_Si'};
    SEDBGC_INI.bgc_frctype = {'i','i','i'} ;
    SEDBGC_INI.bgc_frcini = {BGC_INI.file_bgc{1},BGC_INI.file_bgc{1},BGC_INI.file_bgc{1}} ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if ~exist(chdini)
      disp(['Creating initial file: ' chdini]);
      r2r_create_ini(chdini,chdgrd,chdscd.N,chdscd,makebgc,BGC_INI,bgctracers_list,do_slow)
    end

    r2r_make_ini(pargrd, parscd.file, chdgrd, chdini, chdscd,parscd,parscd.scoord,chdscd.scoord,do_slow)

    if makebgc==1
    r2r_make_inibgc(pargrd, parscd.file, chdgrd, chdini, chdscd,parscd,parscd.scoord,chdscd.scoord,BGC_INI)
       if sed_tracers==1
          r2r_create_sedini(chdini,chdgrd,chdscd.N,chdscd,makebgc,SEDBGC_INI,bgctracers_list)
          r2r_make_inisedbgc(pargrd, parscd.file, chdgrd, chdini, chdscd,parscd,parscd.scoord,chdscd.scoord,SEDBGC_INI)
       end
    end
    






    

