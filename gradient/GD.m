function [x0,i] = GD(f,x,x0,epsilon)
% 梯度下降法 也可以称：最速下降法
% input：f 为上面定义的句柄函数 例如：f= @(x1,x2) 2*x1.^2+2*x2.^2+2*x1.*x2+x1-x2;
%        x 为表示变量字符串，这里一般是 x1,x2
%        x0 表示起始点
%        epsilon 为ε精度
% output：x0 为GD函数出来的从起始点x0寻优到极小值点的所有点集合
%        i 为GD函数寻优过程得到点的个数即：i==length(x0)
%        可以打印出最小值 min{f}，以及极小值值点 x
% @Author     
% Copyright© 2022.5.22 CSDN name: cugautozp

    i=1;
    df = nabla_f(f,x);
    H = Hesse(df,x,x0,i);
    dfx=df(x0{1}(1),x0{1}(2));
   
    er=norm(dfx);
    while er > epsilon                
            p = -dfx;
            lambda = dfx'*dfx/(dfx'*H*dfx);
            i=i+1;
            x0{i} = x0{i-1}+lambda*p;
            dfx = df(x0{i}(1),x0{i}(2));
            H = Hesse(df,x,x0,i);
            er = norm(dfx);     
    end
    fmin = f(x0{i}(1),x0{i}(2));
    disp('极小值点：');
    disp(['x1 = ',num2str(x0{i}(1))]);
    disp(['x2 = ',num2str(x0{i}(2))]);
    fprintf('\nf最小值：\n min{f}=%f\n',fmin);
end



% function H = Hesse(df,x)
% %  Hesse矩阵 H(x)=∇(∇f) 
% %  df 为 ∇f，即为函数nabla_f的结果 ∇f
%     H=[];
%     x=str2sym(x);
%     for i=1:length(x)
%         df1 = diff(df,x(i));
%         H=[H,df1];
%     end
%     H=double(H);
% end

%% 利用subs()函数可以赋值完成任意维度的最速下降法。可以继续修改代码记录时间：2022-05-28