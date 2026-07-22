function [rho,rho1,qp1,rho0] = rho_eos_duko(Tt,Ts,z_r)

%
% Compute density anomaly from T,S via Equation Of State (EOS) for

%-------- ------- ------- ----  for seawater. Following Jackett and 

% McDougall, 1995, physical EOS is assumed to have form
%
%                           rho0 + rho1(T,S)
%           rho(T,S,z) = ------------------------                 (1)
%                         1 - 0.1*|z|/K(T,S,|z|)
%
% where rho1(T,S) is sea-water density perturbation[kg/m^3] at
% standard pressure of 1 Atm (sea surface); |z| is absolute depth,
% i.e. distance from free-surface to the point at which density is
% computed, and
%
%     K(T,S,|z|) = K00 + K01(T,S) + K1(T,S)*|z| + K2(T,S)*|z|^2.  (2)
%
% To reduce errors of pressure-gradient scheme associated with
% nonlinearity of compressibility effects, as well as to reduce
% roundoff errors, the dominant part of density profile,
%
%                           rho0
%                     ----------------                            (3)
%                      1 - 0.1|z|/K00
%
% is removed from from (1). [Since (3) is purely a function of z,
% it does not contribute to pressure gradient.]  This results in
%
%                   rho1 - rho0*[K01+K1*|z|+K2*|z|^2]/[K00-0.1|z|]
%    rho1 + 0.1|z| -----------------------------------------------
%                        K00 + K01 + (K1-0.1)*|z| + K2*|z|^2
%                                                                 (4)
% which is suitable for pressure-gradient calculation.

%
% Optionally, if CPP-switch SPLIT_EOS is defined, term proportional
% to |z| is linearized using smallness 0.1|z|/[K00 + K01] << 1 and
% the resultant EOS has form
%
%              rho(T,S,z) = rho1(T,S) + qp1(T,S)*|z|              (5)
%
% where
%                            rho1(T,S) - rho0*K01(T,S)/K00
%             qp1(T,S)= 0.1 -------------------------------       (6)
%                                   K00 + K01(T,S)
%
% is stored in a special array.
%
% This splitting allows representation of spatial derivatives (and
% also differences) of density as sum of adiabatic derivatives and
% compressible part according to
%
%         d rho     d rho1           d qp1             d |z|
%        ------- = -------- + |z| * -------  +  qp1 * -------     (7)
%         d x,s      d x,s           d x,s             d x,s
%
%                  |<----- adiabatic ----->|   |<- compress ->|
%
% so that constraining of adiabatic derivative for monotonicity is
% equivalent to enforcement of physically stable stratification.
% [This separation and constraining algorithm is subsequently used
% in computation of pressure gradient within prsgrd32ACx-family
% schemes.]
%
% If so prescribed compute the Brunt-Väisäla frequency [1/s^2] at
% horizontal RHO-points and vertical W-points,
%
%                          g    d rho  |
%             bvf^2 = - ------ ------- |                          (8)
%                        rho0    d z   | adiabatic
%
% where density anomaly difference is computed by adiabatically
% rising/lowering the water parcel from RHO point above/below to
% the W-point depth at "z_w".
%
% WARNING: Shared target arrays in the code below: "rho1",
%          "bvf" (if needed), and
%
%          SPLIT_EOS is defined: "qp1"  ["rho" does not exist]
%                   not defined  "rho"  ["qp1" does not exist]
%
%
% Reference:  Jackett, D. R. and T. J. McDougall, 1995, Minimal
%             Adjustment of Hydrostatic Profiles to Achieve Static
%             Stability. J. Atmos. Ocean. Tec., vol. 12, pp. 381-389.
%
% << This equation of state formulation has been derived by Jackett
% and McDougall (1992), unpublished manuscript, CSIRO, Australia. It
% computes in-situ density anomaly as a function of potential
% temperature (Celsius) relative to the surface, salinity (PSU),
% and depth (meters).  It assumes  no  pressure  variation along
% geopotential  surfaces,  that  is,  depth  and  pressure  are
% interchangeable. >>
%                                          John Wilkin, 29 July 92
%

      r00=999.842594;  r01=6.793952E-2; r02=-9.095290E-3; r03=1.001685E-4;  r04=-1.120083E-6;
                                                    r05=6.536332E-9;
      r10=0.824493;     r11=-4.08990E-3;  r12=7.64380E-5;
                        r13=-8.24670E-7;  r14=5.38750E-9;
      rS0=-5.72466E-3;  rS1=1.02270E-4;   rS2=-1.65460E-6;
      r20=4.8314E-4;

      K00=19092.56;     K01=209.8925;     K02=-3.041638;
                        K03=-1.852732e-3; K04=-1.361629e-5;
      K10=104.4077;     K11=-6.500517;    K12=0.1553190;
                                          K13=2.326469e-4;
      KS0=-5.587545;    KS1=+0.7390729;   KS2=-1.909078e-2;


      B00=0.4721788;    B01=0.01028859;   B02=-2.512549e-4;
                                          B03=-5.939910e-7;
      B10=-0.01571896;  B11=-2.598241e-4; B12=7.267926e-6;
                        BS1=2.042967e-3;

      E00=+1.045941e-5; E01=-5.782165e-10;E02=+1.296821e-7;
      E10=-2.595994e-7; E11=-1.248266e-9; E12=-3.508914e-9;

%     rho0 = 1027.439; % Boussinesq background density [kg.m-3]

%     rho0 = 1027.4089895669379;
      %rho0 = 1000;
      rho0 = 1027.4; % rho0 in my ROMS solution

      Tt0 = 3.8e0;
      Ts0 = 34.5e0;

      Ts = max(0,Ts);
      sqrtTs0 =sqrt(Ts0);

      dr00=r00-rho0;

      rho1_0=dr00 +Tt0*( r01+Tt0*( r02+Tt0*( r03+Tt0*( r04+Tt0*r05 ))))     ...
                                  +Ts0*( r10+Tt0*( r11+Tt0*( r12+Tt0*(     ...
                                                    r13+Tt0*r14 )))     ...
                         +sqrtTs0*( rS0+Tt0*( rS1+Tt0*rS2 ))+Ts0*r20 ) ;

      K0_Duk = Tt0*( K01+Tt0*( K02+Tt0*( K03+Tt0*K04 )))  ...
              +Ts0*( K10+Tt0*( K11+Tt0*( K12+Tt0*K13 ))  ...
                   +sqrtTs0*( KS0+Tt0*( KS1+Tt0*KS2 )));


      dr00 = r00 - rho0;

      sqrtTs=sqrt(Ts);

      rho1 = ( dr00 +Tt.*( r01+Tt.*( r02+Tt.*( r03+Tt.*(   ...
                                           r04+Tt*r05 ))))     ...
                         +Ts.*( r10+Tt.*( r11+Tt.*( r12+Tt.*(  ...
                                            r13+Tt*r14 )))     ...
                              +sqrtTs.*(rS0+Tt.*(              ...
                                   rS1+Tt*rS2 ))+Ts*r20 ));

      K0= Tt.*( K01+Tt.*( K02+Tt.*( K03+Tt*K04 )))     ...
         +Ts.*( K10+Tt.*( K11+Tt.*( K12+Tt*K13 ))      ...
              +sqrtTs.*( KS0+Tt.*( KS1+Tt*KS2 )));
 

      qp1 = 0.1*(rho0+rho1).*(K0_Duk-K0)./((K00+K0).*(K00+K0_Duk));


      g = 9.81;
      cff = g/rho0;
      qp2 = 0.0000172;
      rho = rho1 + qp1.*abs(z_r).*(1-qp2*abs(z_r));

      return

      dpth = 0.5*abs(z_r(2:end) + z_r(1:end-1));
      size(rho)
      nz = length(rho);
      bvf = zeros(1,nz+1);
      bvf(2:end-1) = -cff*( rho1(2:end)-rho1(1:end-1) ...
              +(qp1(2:end)-qp1(1:end-1)).*dpth.*(1.- qp2*dpth) ...
                       )./(z_r(2:end)-z_r(1:end-1));


      return
