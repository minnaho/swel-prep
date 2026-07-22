function [z] = zlevs(h,zeta,fname,type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  function z = zlevs(h,zeta,fname,type)
%
%  this function compute the depth of rho or w points for ROMS
%
%  On Input:
%
%    h       bathymetric depth
%    zeta    sea surface height
%    fname   netcdf file that contains relevant Cs_w,Cs_r global attributes
%    type    'r': rho point 'w': w point
%
%  On Output:
%
%    z       Depths (m) of RHO- or W-points (3D matrix).
% 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<4
  error('Not enough input arguments')
end
%
[nx,ny] = size(h);
hc = ncreadatt(fname,'/','hc');

% read Cs coefficients
if type=='w'
  Cs_var = 'Cs_w';
  Cs = ncreadatt(fname,'/','Cs_w');
  nzw = length(Cs);
  nz = nzw-1;
  sc = (0:nz)/nz- 1;
  nz = nzw;
elseif type=='r'
  Cs = ncreadatt(fname,'/','Cs_r');
  nz  = length(Cs);
  sc= (0.5:nz-0.5)/nz-1;
else
  error('unknow z level type (r/w)')
end

%
% Create S-coordinate system: based on model topography h(i,j),
% fast-time-averaged free-surface field and vertical coordinate
% transformation metrics compute evolving depths of of the three-
% dimensional model grid.
%
 z=zeros(nx,ny,nz);
 hinv=1./(h+hc);
 cff=hc*sc;
 cff1=Cs;
 for k=1:nz
    z(:,:,k)=zeta+(zeta+h).*(cff(k)+Cs(k)*h).*hinv;
 end

end
