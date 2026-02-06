function interp_wec(par_grd,par_data,chd_grd,chd_data);
%--------------------------------------------------------------
%
%  Make a roms 2d file for use as WEC forcing using data
%  from a roms WEC forcing parent grid.
%
%  input
%  =====
%  par_grd:         parent grid .nc file name
%  par_data:        parent history .nc file name
%  chd_grd:         child grid .nc file name
%  chd_data:        childt ini .nc file name
%  
%  Minna Ho 2025 at UCLA
%  Heavily modified from French produce
%  Jeroen Molemaker (UCLA); nmolem@atmos.ucla.edu
%--------------------------------------------------------------
%
%
% Get time in parent file
  par_data
  chd_data

  ptime = ncread(par_data,'wwv_time');

% write time in child file
  ncwrite(chd_data,'wwv_time',ptime);

% Get full parent grid and do triangulation
  lonp  = double(ncread(par_grd,'lon_rho')');%/10000;
  latp  = double(ncread(par_grd,'lat_rho')');%/10000;

  [Mpp,Lpp] = size(latp) 
  lonp(lonp<0) = lonp(lonp<0) + 360;

  display('going delaunay');
  tri_fullpar  = delaunay(lonp,latp);
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

% time loop for writing
for tind = 1:size(ptime) 
  disp(['time: ',num2str(tind),' of ',num2str(size(ptime))])
% Do the interpolation for all child chunks
  for domx = 1:ndomx
   for domy = 1:ndomy
     [ domx domy]

      icb = icmin(domx); 
      ice = icmax(domx);
      jcb = jcmin(domy); 
      jce = jcmax(domy);

    % Get topography data from childgrid
      maskc = ncread(chd_grd,'mask_rho',[icb jcb],[ice-icb+1 jce-jcb+1])';
      lonc  = double(ncread(chd_grd,'lon_rho' ,[icb jcb],[ice-icb+1 jce-jcb+1])');
      latc  = double(ncread(chd_grd,'lat_rho' ,[icb jcb],[ice-icb+1 jce-jcb+1])');
      lonc(lonc<0) = lonc(lonc<0) + 360;

      % Compute minimal subgrid extracted from full parent grid
      t = squeeze(tsearch(lonp,latp,tri_fullpar,lonc,latc));

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
      lons(lons<0) = lons(lons<0) + 360;
      if sum(isnan(masks))>0
        disp('Setting NaNs in masks to zero')
        error
        masks(isnan(masks))=0;
        disp('You probably have land masking defined in cppdefs.h...')
      end

      disp('Computing interpolation coefficients');

      %[elem2d,coef2d,nnel] = get_tri_coef(lons,lats,lonc,latc,masks);

    % Open parent data file

      disp('--- Awave')
      Awaves = ncread(par_data,'Awave'  ,[imin jmin tind],[imax-imin+1,jmax-jmin+1 1])';

    % interpolate to child file
      Awaves(masks<1) = nan;
      frc = inpaint_nans(Awaves,2);
      fld = interp2(lons,lats,frc,lonc,latc,'makima',9999);
      Awavec     = fld.*maskc;

      disp('--- Dwave')
      Dwaves = ncread(par_data,'Dwave'  ,[imin jmin tind],[imax-imin+1,jmax-jmin+1 1])';
      Dwaves(masks<1) = nan;
      frc = inpaint_nans(Dwaves,2);
      fld = interp2(lons,lats,frc,lonc,latc,'makima',9999);
      Dwavec     = fld.*maskc;

      disp('--- Pwave')
      Pwaves = ncread(par_data,'Pwave'  ,[imin jmin tind],[imax-imin+1,jmax-jmin+1 1])';
      Pwaves(masks<1) = nan;
      frc = inpaint_nans(Pwaves,2);
      fld = interp2(lons,lats,frc,lonc,latc,'makima',9999);
      Pwavec     = fld.*maskc;

      disp('--- uorb')
      uorbs = ncread(par_data,'uorb'  ,[imin jmin tind],[imax-imin+1,jmax-jmin+1 1])';
      uorbs(masks<1) = nan;
      frc = inpaint_nans(uorbs,2);
      fld = interp2(lons,lats,frc,lonc,latc,'makima',9999);
      uorbc     = fld.*maskc;

      disp('--- vorb')
      vorbs = ncread(par_data,'vorb'  ,[imin jmin tind],[imax-imin+1,jmax-jmin+1 1])';
      vorbs(masks<1) = nan;
      frc = inpaint_nans(vorbs,2);
      fld = interp2(lons,lats,frc,lonc,latc,'makima',9999);
      vorbc     = fld.*maskc;

      disp('--- ust2d')
      ust2ds = ncread(par_data,'ust2d'  ,[imin jmin tind],[imax-imin+1,jmax-jmin+1 1])';
      ust2ds(masks<1) = nan;
      frc = inpaint_nans(ust2ds,2);
      fld = interp2(lons,lats,frc,lonc,latc,'makima',9999);
      ust2dc     = fld.*maskc;

      disp('--- vst2d')
      vst2ds = ncread(par_data,'vst2d'  ,[imin jmin tind],[imax-imin+1,jmax-jmin+1 1])';
      vst2ds(masks<1) = nan;
      frc = inpaint_nans(vst2ds,2);
      fld = interp2(lons,lats,frc,lonc,latc,'makima',9999);
      vst2dc     = fld.*maskc;

      disp('--- ust0')
      ust0s = ncread(par_data,'ust0'  ,[imin jmin tind],[imax-imin+1,jmax-jmin+1 1])';
      ust0s(masks<1) = nan;
      frc = inpaint_nans(ust0s,2);
      fld = interp2(lons,lats,frc,lonc,latc,'makima',9999);
      ust0c     = fld.*maskc;

      disp('--- vst0')
      vst0s = ncread(par_data,'vst0'  ,[imin jmin tind],[imax-imin+1,jmax-jmin+1 1])';
      vst0s(masks<1) = nan;
      frc = inpaint_nans(vst0s,2);
      fld = interp2(lons,lats,frc,lonc,latc,'makima',9999);
      vst0c     = fld.*maskc;

      disp('--- ed')
      eds = ncread(par_data,'ed'  ,[imin jmin tind],[imax-imin+1,jmax-jmin+1 1])';
      eds(masks<1) = nan;
      frc = inpaint_nans(eds,2);
      fld = interp2(lons,lats,frc,lonc,latc,'makima',9999);
      edc     = fld.*maskc;

      disp('--- eb')
      ebs = ncread(par_data,'eb'  ,[imin jmin tind],[imax-imin+1,jmax-jmin+1 1])';
      ebs(masks<1) = nan;
      frc = inpaint_nans(ebs,2);
      fld = interp2(lons,lats,frc,lonc,latc,'makima',9999);
      ebc     = fld.*maskc;

      disp('--- qb')
      qbs = ncread(par_data,'qb'  ,[imin jmin tind],[imax-imin+1,jmax-jmin+1 1])';
      qbs(masks<1) = nan;
      frc = inpaint_nans(qbs,2);
      fld = interp2(lons,lats,frc,lonc,latc,'makima',9999);
      qbc     = fld.*maskc;

      disp('--- sup')
      sups = ncread(par_data,'sup'  ,[imin jmin tind],[imax-imin+1,jmax-jmin+1 1])';
      sups(masks<1) = nan;
      frc = inpaint_nans(sups,2);
      fld = interp2(lons,lats,frc,lonc,latc,'makima',9999);
      supc     = fld.*maskc;

      disp('--- lmw')
      lmws = ncread(par_data,'lmw'  ,[imin jmin tind],[imax-imin+1,jmax-jmin+1 1])';
      lmws(masks<1) = nan;
      frc = inpaint_nans(lmws,2);
      fld = interp2(lons,lats,frc,lonc,latc,'makima',9999);
      lmwc     = fld.*maskc;

      
      disp(' Writing ini file')
      ncwrite(chd_data,'Awave'  ,Awavec'  ,[icb jcb tind]  );
      ncwrite(chd_data,'Dwave'  ,Dwavec'  ,[icb jcb tind]  );
      ncwrite(chd_data,'Pwave'  ,Pwavec'  ,[icb jcb tind]  );
      ncwrite(chd_data,'uorb'  ,uorbc'  ,[icb jcb tind]  );
      ncwrite(chd_data,'vorb'  ,vorbc'  ,[icb jcb tind]  );
      ncwrite(chd_data,'ust2d'  ,ust2dc'  ,[icb jcb tind]  );
      ncwrite(chd_data,'vst2d'  ,vst2dc'  ,[icb jcb tind]  );
      ncwrite(chd_data,'ust0'  ,ust0c'  ,[icb jcb tind]  );
      ncwrite(chd_data,'vst0'  ,vst0c'  ,[icb jcb tind]  );
      ncwrite(chd_data,'ed'  ,edc'  ,[icb jcb tind]  );
      ncwrite(chd_data,'eb'  ,ebc'  ,[icb jcb tind]  );
      ncwrite(chd_data,'qb'  ,qbc'  ,[icb jcb tind]  );
      ncwrite(chd_data,'sup'  ,supc'  ,[icb jcb tind]  );
      ncwrite(chd_data,'lmw'  ,lmwc'  ,[icb jcb tind]  );

   end
  end
end
