function H = Hesse(df,x,x0,n)
%  Hesse矩阵 H(x)=∇(∇f) 
%  df 为 ∇f，即为函数nabla_f的结果 ∇f
% @Author     
% Copyright© 2022.5.22 CSDN name: cugautozp
    H=[];
    x=str2sym(x);
    for i=1:length(x)
        df1 = diff(df,x(i));
        H=[H,df1];
    end
%     H = matlabFunction(H);
    s=char(H);
    if find(s=='x')
        H = matlabFunction(H);
        H = H(x0{n}(1),x0{n}(2));
    else
        H = double(H);
    end
end