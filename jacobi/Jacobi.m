function varargout=Jacobi(f,varargin)
% 求函数表达式雅可比矩阵 J
% 输出：
%       输出1：符号表达式
%       输出2：句柄函数
%       输出3：函数 f 的变量(以元胞数组(cell)形式存放中，n × 1维)
%       输出4：若输入点x0, 输出带入点后的值
% @Author     
% Copyright© 2022.10.22 CSDN name: cugautozp

    [x,f]=fx(f);
    n=nargin(f); % 找到输入参数个数
    df=[];        
    for i =1:n
        df1 = diff(f,x(i));
        df = [df,df1];
    end
%     J=matlabFunction(df);    
    varargout{1}=df;                       % 输出为符号表达式
    varargout{2}=matlabFunction(df);       % 输出为句柄函数
    for i=1:length(x)
            s{i}=char(x(i));
    end
    varargout{3}=s;                        % 输出变量
    if ~isempty(varargin)
        varargout{4}=Jx(df,s,varargin{1}); % 输出代入点后的值 
    end
end

function [x,f]=fx(f)
% 将用字符串写的函数表达式转化为句柄函数
% 输入: f 为函数表达式(字符串形式 / 符号函数)
% 输出: x 为函数 f 中的变量
    if  ~isa(f,'sym')            % 判断f是否为符号函数格式。
        if iscolumn(f)
            f=str2sym(f);
        else
            f=str2sym(f');
        end                 
    end 
    x=symvar(f);             % 搜寻函数中的符号变量
    f=matlabFunction(f);
end

function Jk=Jx(J,x,x0)
% 将点 x0 代入雅可比矩阵 J 中求值
% 输出格式为：矩阵值
    n=nargin(matlabFunction(J));
    if n==0
        Jk=double(J);
    else
        a=symvar(J);  % 找雅可比矩阵中的符号变量
        for i=1:length(a)
            s=char(a(i));
            idx(i) = find(strcmp(x,s));
        end
        Jk = subs(J,a,x0(idx));
        Jk = double(Jk);
    end   
end