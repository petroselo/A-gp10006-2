%Uniform Flow Characteristics
U=10;
a=0;

%Sourc/Sink Characteristics
qs=1;
Qs=qs/(2*pi);

%Doublet Characteristics
qd=10000000;
Qd=qd/(2*pi);
e=0.01;

%Vortex Characteristics
circ=1;

%Locations
sx=0;
sy=0;
dx=0;
dy=0;
vx=0;
vy=0;

%Grid Sizing
size=10;
grid_space=1;

reso=-1*size:grid_space:size;
[x,y]=meshgrid(reso,reso);

z=x+1i*y;

Fu=U*(cos((a*(pi/180)))-1i*sin((a*(pi/180))))*(z);

Fs=Qs*log(z-sx-sy*1i);

Fd=Qd*log(z-dx-dy*1i+e)-Qd*log(z-dx-dy*1i-e);

Fv=-1i*circ*log(z-vx-1i*vy);

%F=Fu+Fs+Fd+Fv;
F = Fd + Fu;

%contour(x,y,imag(F),-100:10:100);
contour(x,y,imag(F));

%hold on;
%contour(x,y,real(F),-100:10:100);



axis square;