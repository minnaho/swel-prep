% correction_on_ini

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%  2019, Pierre Damien (UCLA)
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 clear all
 close all


%%%%%%%%%%%%%%%%%%%%% USER-DEFINED VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%
%
 
 wrk_dir     = '/data/project3/pdamien/ROMS_pdamien/config/pacmed12km/grid/';
 frc_dir     = '/data/project3/pdamien/ROMS_pdamien/config/pacmed12km/out_interannual/'; 
 grdname     = [wrk_dir, 'pacmed_12km_grd.nc'];
 frcname     = [frc_dir, 'pacmed_12km_ini2000_roms25km.nc'];

%
%%%%%%%%%%%%%%%%%%% END USER-DEFINED VARIABLES %%%%%%%%%%%%%%%%%%%%%%%
%

disp(' ')
disp('correcting ubar')  
frcval = ncread( frcname , 'ubar',[1 1 1],[Inf Inf Inf]) ;
frcval(isnan(frcval)) = 0 ;
ncwrite(frcname , 'ubar' , frcval ,[1 1 1]) ;
clear frcval
disp(' ')

disp(' ')
disp('correcting vbar')  
frcval = ncread( frcname , 'vbar',[1 1 1],[Inf Inf Inf]) ;
frcval(isnan(frcval)) = 0 ;
ncwrite(frcname , 'vbar' , frcval ,[1 1 1]) ;
clear frcval
disp(' ')

disp(' ')
disp('correcting zeta')  
frcval = ncread( frcname , 'zeta',[1 1 1],[Inf Inf Inf]) ;
frcval(isnan(frcval)) = 0 ;
ncwrite(frcname , 'zeta' , frcval ,[1 1 1]) ;
clear frcval
disp(' ')

disp(' ')
disp('correcting u')  
frcval = ncread( frcname , 'u',[1 1 1 1],[Inf Inf Inf Inf]) ;
frcval(isnan(frcval)) = 0 ;
ncwrite(frcname , 'u' , frcval ,[1 1 1 1]) ;
clear frcval
disp(' ')

disp(' ')
disp('correcting v')  
frcval = ncread( frcname , 'v',[1 1 1 1],[Inf Inf Inf Inf]) ;
frcval(isnan(frcval)) = 0 ;
ncwrite(frcname , 'v' , frcval ,[1 1 1 1]) ;
clear frcval
disp(' ')

disp(' ')
disp('correcting w')  
frcval = ncread( frcname , 'w',[1 1 1 1],[Inf Inf Inf Inf]) ;
frcval(isnan(frcval)) = 0 ;
ncwrite(frcname , 'w' , frcval ,[1 1 1 1]) ;
clear frcval
disp(' ')

disp(' ')
disp('correcting temp')  
frcval = ncread( frcname , 'temp',[1 1 1 1],[Inf Inf Inf Inf]) ;
frcval(isnan(frcval)) = 0 ;
ncwrite(frcname , 'temp' , frcval ,[1 1 1 1]) ;
clear frcval
disp(' ')

disp(' ')
disp('correcting salt')  
frcval = ncread( frcname , 'salt',[1 1 1 1],[Inf Inf Inf Inf]) ;
frcval(isnan(frcval)) = 0 ;
ncwrite(frcname , 'salt' , frcval ,[1 1 1 1]) ;
clear frcval
disp(' ')

