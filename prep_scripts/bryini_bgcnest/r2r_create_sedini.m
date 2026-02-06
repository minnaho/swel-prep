function r2r_create_sedini(inifile,gridfile,N,chdscd,makebgc,BGC_INI,bgctracers_list)
%
%   Input: 
% 
%   inifile      Netcdf initial file name (character string).
%   gridfile     Netcdf grid file name (character string).
%   clobber      Switch to allow writing over an existing
%                file (character string)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%  Read the grid file
%
h       = ncread(gridfile,'h')';
[Mp,Lp] = size(h);
L       = Lp - 1 ;
M       = Mp - 1 ;
Np      = N  + 1 ;

%
%
%  Create variables and attributes
%
    
    for trc=1:length(BGC_INI.bgc_tracer)
        nccreate(inifile,BGC_INI.bgc_tracer{trc},'Dimensions',{'xi_rho',Lp,'eta_rho',Mp,'time',1},'datatype','single');
    end

return


