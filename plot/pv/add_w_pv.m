 gname = '/zulu/nmolem/SRRIDGE/FRC/srridge_grd.nc';
 path  = '/zulu/nmolem/SRRIDGE/HIS/';
 list = dir([path 'srridge_his.2*']);
 nfiles = length(list);

 h = ncread(gname,'h');
 f = ncread(gname,'f');
 dx = 1./ncread(gname,'pm');
 dy = 1./ncread(gname,'pn');

 dxu = 0.5*(dx(2:end,2:end-1)+dx(1:end-1,2:end-1));
 dxv = 0.5*(dx(2:end-1,2:end)+dx(2:end-1,1:end-1));
 dyu = 0.5*(dy(2:end,2:end-1)+dy(1:end-1,2:end-1));
 dyv = 0.5*(dy(2:end-1,2:end)+dy(2:end-1,1:end-1));

 [nx,ny] = size(dx);
 nxu = nx-1;
 nyv = ny-1;

 fname = [path list(1).name];
 nz = length(squeeze(ncread(fname,'u',[1 1 1 1],[1 1 inf 1]) ));
 
 for i = 25:nfiles

   fname = [path list(i).name] 
   tim = ncread(fname,'ocean_time');
   nfr = length(tim);
   try
     nccreate(fname,'w','dimensions',{'xi_rho',nx,'eta_rho',ny,'s_rho',nz,'time',0},'datatype','single');
   catch
     disp('w already exists')
   end
   ncwriteatt(fname,'w','long_name','w velocity');
   ncwriteatt(fname,'w','units','m/s');

   try 
     nccreate(fname,'pv','dimensions',{'xi_rho',nx,'eta_rho',ny,'s_rho',nz,'time',0},'datatype','single');
   catch
     disp('pv already exists')
   end
   ncwriteatt(fname,'pv','long_name','Potential vorticity');
   ncwriteatt(fname,'pv','units','m/s3');


   for ifr=1:nfr;
     ifr
     u = ncread(fname,'u',[1 1 1 ifr],[inf inf inf 1]);
     v = ncread(fname,'v',[1 1 1 ifr],[inf inf inf 1]);

     z = ncread(fname,'zeta',[1 1 ifr],[inf inf 1]);
     zr = zlevs(h,z,fname,'r');
     zw = zlevs(h,z,fname,'w');
     dz = zw(:,:,2:end)-zw(:,:,1:end-1);

%    w = calc_w(u,v,zr,dz,dx,dy);
%    ncwrite(fname,'w',w,[1 1 1 ifr]) ;

     temp = ncread(fname,'temp',[1,1,1,ifr],[inf inf inf 1]);
     salt = ncread(fname,'salt',[1,1,1,ifr],[inf inf inf 1]);

     pv = calc_pv(u,v,temp,salt,zr,dz,dx,dy,f);
     return

     ncwrite(fname,'pv',pv,[1 1 1 ifr]);


   end
 end
