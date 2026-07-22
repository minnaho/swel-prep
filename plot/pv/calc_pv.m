function pv = calc_pv(u,v,temp,salt,zr,dz,dx,dy,f)

     [nx,ny,nz] = size(zr);
     dx = repmat(dx,[1 1 nz]);
     dy = repmat(dy,[1 1 nz]);

     g = 9.81;
     qp2=0.0000172;

     % gradients of density field
     [rho,rho1,qp1,rho0] = rho_eos_duko(temp,salt,zr);
     if 1
%     [bx,by,bz] = rho_grad(rho1,qp1,qp2,zr);
      [bx,by,bz] = rho_grad(rho1);
      bx = -g*bx./dx/rho0;
      by = -g*by./dy/rho0;
      bz = -g*bz./dz/rho0;
     else
      b = -g*rho1/rho0;
      [by,bx,bz] = gradient(b);
      bx = bx./dx;
      by = by./dy;
      bz = bz./dz;
     end

     u = u2rho(u);
     v = v2rho(v);
     [uy, ~,uz] = gradient(u);
     [ ~,vx,vz] = gradient(v);
     vx = vx./dx;
     uy = uy./dy;

     uz = uz./dz;
     vz = vz./dz;

%    Ro = (vx - uy)./f;
%    S2 = uz.^2 + vz.^2;
%    [nx,ny,nz] = size(pv);

     pv = -vz.*bx + uz.*by + (f + vx - uy).*bz;
end
