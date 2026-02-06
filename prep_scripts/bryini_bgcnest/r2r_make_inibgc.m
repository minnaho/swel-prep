function r2r_make_inibgc(par_grd,par_data,chd_grd, chd_data,   ...
                      chdscd,   parscd, scoord_switch_p, scoord_switch_c , BGC_INI);
%--------------------------------------------------------------
%
%  Make a roms 3d file for use as an initial file using data
%  from a roms parent grid.
%
%  input
%  =====
%  par_grd:         parent grid .nc file name
%  par_data:        parent history .nc file name
%  chd_grd:         child grid .nc file name
%  chd_data:        childt ini .nc file name
%  chdscd:          child grid s-coordinate parameters (object)
%  parscd:          parent grid s-coordinate parameters (object)
%  scoord_switch_c: child 'old' or 'new' s-coordinate
%  scoord_switch_p: parent 'old' or 'new' s-coordinate
%
%  Heavily modified from French produce
%  Jeroen Molemaker (UCLA); nmolem@atmos.ucla.edu
%--------------------------------------------------------------
%
%
% Get S-coordinate params for child grid
  N_c       = chdscd.N;
  theta_b_c = chdscd.theta_b;
  theta_s_c = chdscd.theta_s;
  hc_c      = chdscd.hc;

% Get S-coordinate params for parent grid
  N_p       = parscd.N;
  theta_b_p = parscd.theta_b;
  theta_s_p = parscd.theta_s;
  hc_p      = parscd.hc;
  par_file  = parscd.file;

% Get S-coordinate params for parent data file
  tind = parscd.tind;

% Set correct time in ini file
% np = netcdf(par_data, 'nowrite');
% ni = netcdf(chd_data, 'write');
  par_data
  chd_data
%   ncread(par_data,'time',tind,1)/(24*3600)
   ptime = ncread(par_data,'ocean_time',tind,1);
% % ni{'ocean_time'}(:) = np{'ocean_time'}(tind) + 100;  %% Adding 100 seconds to fix rounding problems in bry_time.
   ncwrite(chd_data,'ocean_time',ptime);
%  ncwrite(chd_data,'ocean_time',0);

% Get full parent grid and do triangulation
  lonp  = double(ncread(par_grd,'lon_rho')');%/10000;
  latp  = double(ncread(par_grd,'lat_rho')');%/10000;

  [Mpp,Lpp] = size(latp) 
  lonp(lonp<0) = lonp(lonp<0) + 360;

  display('going delaunay');
 tri_fullpar = delaunay(lonp,latp);
%  tri_fullpar = DelaunayTri([reshape(lonp,Mpp*Lpp,1),reshape(latp,Mpp*Lpp,1)]);
  display('return delaunay');

% Get child grid and chunk size
  ndomx = 1;
  ndomy = 2;
  [Mp,Lp] = size(ncread(chd_grd,'h')');
  szx = floor(Lp/ndomx);
  szy = floor(Mp/ndomy);

  icmin = [0:ndomx-1]*szx;
  jcmin = [0:ndomy-1]*szy;
  icmax = [1:ndomx]*szx;
  jcmax = [1:ndomy]*szy;
  icmin(1) = 1;
  jcmin(1) = 1;
  icmax(end) = Lp;
  jcmax(end) = Mp;

% Do the interpolation for all child chunks
  for domx = 1:ndomx
   for domy = 1:ndomy
     [ domx domy]

      icb = icmin(domx); 
      ice = icmax(domx);
      jcb = jcmin(domy); 
      jce = jcmax(domy);

    % Get topography data from childgrid
%     hc1 = ncread(chd_grd,'h')'; hc1 = hc1(jcb:jce,icb:ice);
      hc    = ncread(chd_grd,'h'       ,[icb jcb],[ice-icb+1 jce-jcb+1])';
      maskc = ncread(chd_grd,'mask_rho',[icb jcb],[ice-icb+1 jce-jcb+1])';
      angc  = ncread(chd_grd,'angle'   ,[icb jcb],[ice-icb+1 jce-jcb+1])';
%     lonc  = ncread(chd_grd,'lon_rho' ,[icb jcb],[ice-icb+1 jce-jcb+1])'/10000;
%     latc  = ncread(chd_grd,'lat_rho' ,[icb jcb],[ice-icb+1 jce-jcb+1])'/10000;
      lonc  = double(ncread(chd_grd,'lon_rho' ,[icb jcb],[ice-icb+1 jce-jcb+1])');
      latc  = double(ncread(chd_grd,'lat_rho' ,[icb jcb],[ice-icb+1 jce-jcb+1])');
      umask = maskc(:,1:end-1).*maskc(:,2:end);
      vmask = maskc(1:end-1,:).*maskc(2:end,:);
      cosc  = cos(angc);
      sinc  = sin(angc);
      lonc(lonc<0) = lonc(lonc<0) + 360;
%     figure
%     plot(lonp,latp,'.k');
%     hold on
%     plot(lonc(1,:),latc(1,:),'.r');
%     plot(lonc(:,end),latc(:,end),'.r');
%     plot(lonc(end,:),latc(end,:),'.r');
%     plot(lonc(:,1),latc(:,1),'.r');
%     hold off
%     error 'testing'

    % Compute minimal subgrid extracted from full parent grid
    t = squeeze(tsearch(lonp,latp,tri_fullpar,lonc,latc));
%       [nyc,nxc] = size(lonc);
%       t   = squeeze(pointLocation(tri_fullpar,reshape(lonc,nxc*nyc,1),reshape(latc,nxc*nyc,1)));
%       sum(isnan(t))

    % Deal with child points that are outside parent grid (those points should be masked!)
      if (length(t(~isfinite(t)))>0);
       disp('Warning in new_bry_subgrid: outside point(s) detected.');
       [lonc,latc] = fix_outside_child(lonc,latc,t);
       t = squeeze(tsearch(lonp,latp,tri_fullpar,lonc,latc));
%       t = squeeze(pointLocation(tri_fullpar,reshape(double(lonc),nxc*nyc,1),reshape(double(latc),nxc*nyc,1)));
      end;
      index       = tri_fullpar(t,:);
      [idxj,idxi] = ind2sub([Mpp Lpp], index);

      imin = min(min(idxi));% imin = max(1,imin-1);
      imax = max(max(idxi));% imax = min(1,imin-1);
      jmin = min(min(idxj));
      jmax = max(max(idxj));

    % Get parent grid and squeeze minimal subgrid
      [imin imax jmin jmax]
      par_grd
      masks = ncread(par_grd,'mask_rho',[imin jmin],[imax-imin+1,jmax-jmin+1])';
      lons  = ncread(par_grd,'lon_rho' ,[imin jmin],[imax-imin+1,jmax-jmin+1])'; %lons = double(lons)/1e4;
      lats  = ncread(par_grd,'lat_rho' ,[imin jmin],[imax-imin+1,jmax-jmin+1])'; %lats = double(lats)/1e4;
      angs  = ncread(par_grd,'angle'   ,[imin jmin],[imax-imin+1,jmax-jmin+1])';
      hs    = ncread(par_grd,'h'       ,[imin jmin],[imax-imin+1,jmax-jmin+1])';
      lons(lons<0) = lons(lons<0) + 360;
      coss = cos(angs); sins = sin(angs);
      if sum(isnan(masks))>0
        disp('Setting NaNs in masks to zero')
        error
        masks(isnan(masks))=0;
        disp('You probably have land masking defined in cppdefs.h...')
      end
% 
%       figure
%       plot(lons,lats,'.k')
%       hold on
%       plot(lonc,latc,'.r')
%       hold off
%       return
%     size(hs)
    % Z-coordinate (3D) on minimal subgrid and child grid
      zs = zlevs4(hs, hs*0, theta_s_p, theta_b_p, hc_p, N_p, 'r', scoord_switch_p);
%     zs = zlev_cf(hs, hs*0, par_file, 'r');
      zc = zlevs4(hc, hc*0, theta_s_c, theta_b_c, hc_c, N_c, 'r', scoord_switch_c);
      zw = zlevs4(hc, hc*0, theta_s_c, theta_b_c, hc_c, N_c, 'w', scoord_switch_c);

      [Np Mp Lp] = size(zs);
      [Nc Mc Lc] = size(zc);

      disp('Computing interpolation coefficients');
%     lonc(lonc<0) = lonc(lonc<0) + 360;
%     plot(lons,lats,'.k');
%     hold on
%     plot(lonc(1,1),latc(1,1),'.r');
%       plot(lonc(1,end),latc(1,end),'.r');
%       plot(lonc(end,1),latc(end,1),'.r');
%       plot(lonc(end,end),latc(end,end),'.r');
%     hold off
%     error 'testing'
      [elem2d,coef2d,nnel] = get_tri_coef(lons,lats,lonc,latc,masks);
      A = get_hv_coef(zs, zc, coef2d, elem2d, lons, lats, lonc, latc);

    % Open parent data file
      clear vars vars_new input_var_name
      for trc=1:length(BGC_INI.tracer_roms)
%      filefrc = BGC_INI.bgc_frcini{BGC_INI.tracer_roms(trc)} ; 
      trcname = BGC_INI.bgc_tracer{BGC_INI.tracer_roms(trc)} ;
      disp(['--- ' trcname])
      if strcmp(BGC_INI.bgc_frctype(trc),'i')==1
      var = squeeze(ncread(par_data,trcname  ,[imin jmin 1 tind],[imax-imin+1,jmax-jmin+1 inf 1]));
      var = permute(var,[3 2 1]);
      var = fillmask(var,1,masks,nnel);
      var = double(var);
      ini_var = reshape(A*reshape(var,Np*Mp*Lp,1),Nc,Mc,Lc);

    % Zero-ing out the mask
      for k = 1:Nc
        ini_var(k,:,:) = squeeze(ini_var(k,:,:)); %.*maskc;
      end
      
      disp(' Writing ini file , for netcdf')
      ini_var = permute(ini_var,[3 2 1]);
      ncwrite(chd_data,trcname  ,ini_var,[icb jcb 1 1]);
      elseif strcmp(BGC_INI.bgc_frctype(trc),'v')==1
      disp(' Writing ini file , value')
      ini_var = ini_var.*0+BGC_INI.bgc_frcini{trc} ; 
      ncwrite(chd_data,trcname  ,ini_var,[icb jcb 1 1]);
      end
      end

   end
  end

  disp('----- > Variables Scaled ')

   vec = strfind(BGC_INI.bgc_frctype,'s') ;
   for trc=1:length(vec)
       if vec{trc}==1
       scaled_tracer = BGC_INI.bgc_tracer{trc} ;
       ref_ind       = str2num(BGC_INI.bgc_frctype{trc}(3:end)) ;
       tracer_scaled = BGC_INI.bgc_tracer{ref_ind} ;
       factor        = str2num(BGC_INI.bgc_frcini{trc}) ;
       disp(['SCALING variable ' scaled_tracer ' from ' tracer_scaled ' and factor ' num2str(factor)])
       var = ncread(chd_data,tracer_scaled) ;
       ncwrite(chd_data,scaled_tracer,var*factor,[1 1 1 1]);
       end
   end






