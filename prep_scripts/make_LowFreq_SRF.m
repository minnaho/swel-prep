%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% make_LowFreq_SRF
%
% This script reads surface forcings and make a LowPass filtered
% version. A use is to filter the wind to avoid the NIW injection, 
% or filter the SW radiation to constrain nitrification in BGC runs.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
disp(' ')
%---------------------------------------------------------------------------------------
%  USER-DEFINED VARIABLES & OPTIONS START HERE
%---------------------------------------------------------------------------------------
    romsdir    = '/data/project9/minnaho/swel/tides/mc60/';
    grdname    = [romsdir 'mc60_grd.nc'];
    inname     = [romsdir 'frc/mc60_frc.'];
    outname    = [romsdir 'frc/mc60_LFfrc_'];
    start_date = datenum(2019,04,01);
    end_date   = datenum(2019,05,31);
    vars = {'swrad'} ; 
    time = {'rad_time'} ;
    filt = 72 ; % filtering windows
%---------------------------------------------------------------------------------------
% USER-DEFINED VARIABLES & OPTIONS END HERE
%---------------------------------------------------------------------------------------

% make one bry file per year
  start_year = str2num(datestr(start_date,'YYYY')) ;
  end_year = str2num(datestr(end_date,'YYYY')) ;

for yy=start_year:end_year

    disp(['Working on year ' num2str(yy)])
    list_in = dir([inname num2str(yy) '*.nc']);
    file_out = [outname num2str(yy) '.nc'] ;

    for v=1:length(vars)
        clear timLF varLF 
        ind = 0 ; 
        for t=1:length(list_in)
            file = [list_in(t).folder '/' list_in(t).name] ;
            timHF = ncread(file,time{v}) ; 
            nb = floor(length(timHF)/filt) ; istr = 1 ;
            for i = 1:nb
                ind=ind+1 ; 
                timLF(ind) = mean(timHF(istr:istr+filt-1)) ; 
                varLF(:,:,ind) = squeeze(mean(ncread(file,vars{v},[1 1 istr],[inf inf filt]),3)) ; 
                istr = istr+filt ; 
            end
        end
        [nx,ny,nt] = size(varLF) ; 
        nccreate(file_out,[time{v} '_LFreq'],'Dimensions',{'time',nt},'datatype','single');
        varat = ncreadatt(file,time{v},'long_name') ;
        ncwriteatt(file_out,[time{v} '_LFreq'],'long_name',['Low-freq (window=' num2str(filt)  ') ' varat]);
        varat = ncreadatt(file,time{v},'units') ;
        ncwriteatt(file_out,[time{v} '_LFreq'],'units',varat);
        ncwrite(file_out,[time{v} '_LFreq'],timLF) ;
        nccreate(file_out,[vars{v} '_LFreq'],'Dimensions',{'xi_rho',nx,'eta_rho',ny,'time',nt},'datatype','single');
        varat = ncreadatt(file,vars{v},'long_name') ;
        ncwriteatt(file_out,[vars{v} '_LFreq'],'long_name',['Low-freq (window=' num2str(filt)  ') ' varat]);
        varat = ncreadatt(file,vars{v},'units') ;
        ncwriteatt(file_out,[vars{v} '_LFreq'],'units',varat);
        ncwrite(file_out,[vars{v} '_LFreq'],varLF) ;
    end       

end


