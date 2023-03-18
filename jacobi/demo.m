clc
clear
% f={'x^2+y^2+z^2','y^z*cos(x)','x^z*sin(y)','x^(y^z)'};
% [J,Jf,var] = Jacobi(f)

syms x y z
f = [z*exp(x^y);x;z;y]
J = Jacobi(f)
j1 = jacobian(f,[x,y,z])