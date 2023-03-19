function s = f2s(fun)
% 句柄函数的转换为字符串 
% 主要是用来画图title用的，可自动将句柄函数转为字符串函数
% 在绘制画图时可以自动生成函数表达式 title（并且以Latex形式显示出来），避免手动敲击函数公式
    s = func2str(fun);
    s = char(s);
    c = strfind(s,')');
    s(1:c(1))=[];
    c1 = strfind(s,'.');
    s(c1)=[];
    c2 = strfind(s,'*');
    s(c2)=[];
    c3 = strfind(s,'x');
    for  i = 1:length(c3)
       s = insertAfter(s,c3(i)+i-1,'_'); 
    end   
end