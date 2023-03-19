clc
clear
close all
% f= @(x1,x2) 2*x1.^2+2*x2.^2+2*x1.*x2+x1-x2; % 示例1
f= @(x1,x2) x1.^2+2*x2.^2-2*x1.*x2-2*x2;      % 句柄函数表达式，示例2
% f= @(x1,x2) (x1-1).^2+(x2-1).^2;            % 示例3
% f=@(x1,x2) x1.^4+3*x1.^2*x2-x2.^4;          % 示例4
x0{1}=[0;0];       % 起始点
x ='[x1,x2]';      % 函数变量字符串
epsilon=0.001;     % ε为误差精度
[x0,i] = GD(f,x,x0,epsilon);
x1=0:0.01:2;
x2=x1;
GIFname = 'f1.gif';
GDplot(f,x0,i,x1,x2,GIFname)








