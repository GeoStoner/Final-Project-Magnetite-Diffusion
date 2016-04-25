% FTCS diffusion model to model arctic temperatures
% Modified by Ryan Stoner jan 25 2016

clf

N = 50; % Number of dz steps
dz = 0.5;  % spacing in m
z = 0:dz:(N-1)*dz;
k = 2;   % W/mK
dt = 2.00*10^5;  % s


time= 1:dt:10^7; % All time steps
tmax= length(time);

rho = 2000; % kg/m3
Cp = 2000; % J/kg K

T = zeros(N,1);
dTdz = 0.025* ones(N,1);
q = zeros(N,1);

T(1) = -10;  % brr! -10 C
dTdz(N) = 0.025;  % K/m


for i = 1:tmax;

% put following in loop

% calculate T gradient
dTdz(1:N-1) = diff(T)/dz;

% calculate heat flux
q = -k*dTdz;

% rate of change of T
dqdz = diff(q)/dz;

% update T
T(2:N) = T(2:N) - (1/(rho*Cp))*dqdz*dt;
pause(0.01)
figure(1)
plot(transpose(T),z,'k-')
set(gca,'Ydir','reverse')

end
