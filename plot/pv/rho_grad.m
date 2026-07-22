function [rx,ry,rz] = rho_grad(rho1,qp1,qp2,zr);
%      rz(i,k)=rho1(i,j,k+1)-rho1(i,j,k) 
%    &  +(qp1(i,j,k+1)-qp1(i,j,k))*dpth*(1.-qp2*dpth)

       if nargin < 4
         do_qp = 0;
       else
         do_qp = 1;
         disp('adding qp effects')
       end

       [nx,ny,nz] = size(rho1);

       rzw = rho1(:,:,2:end)-rho1(:,:,1:end-1);
       if do_qp
        dpth = -0.5*(zr(:,:,2:end)+zr(:,:,1:end-1));
        rzw = rzw + ...
           (qp1(:,:,2:end)- qp1(:,:,1:end-1)).*dpth.*(1.-qp2*dpth);
       end

       rz = zeros(nx,ny,nz);
       rz(:,:,2:end-1) = 0.5*( rzw(:,:,2:end)+ rzw(:,:,1:end-1));
       rz(:,:,1 ) = rzw(:,:,2);
       rz(:,:,nz) = rzw(:,:,nz-1);

       rx = rho1(2:end,:,:)-rho1(1:end-1,:,:);
       if do_qp
        dpth = -0.5*(zr(2:end,:,:)+zr(1:end-1,:,:));
        rx = rx + ...
          (qp1(2:end,:,:)- qp1(1:end-1,:,:)).*dpth.*(1.-qp2*dpth);
       end
       rx = u2rho(rx);

       ry = rho1(:,2:end,:)-rho1(:,1:end-1,:);
       if do_qp
        dpth = -0.5*(zr(:,2:end,:)+zr(:,1:end-1,:));
        ry = ry + ...
          (qp1(:,2:end,:)-qp1(:,1:end-1,:)).*dpth.*(1.-qp2*dpth);
       end
       ry = v2rho(ry);




