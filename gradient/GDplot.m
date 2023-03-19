function GDplot(f,x0,i,x1,x2,GIFname)
% input：f 为上面定义的句柄函数 例如：f= @(x1,x2) 2*x1.^2+2*x2.^2+2*x1.*x2+x1-x2;
%        x0 为GD函数出来的从起始点x0寻优到极小值点的所有点集合
%        i 为GD函数寻优过程得到点的个数即：i==length(x0)
%        x1 为f函数中第一个变量的取值范围
%        x2 为f函数中第二个变量的取值范围
% 
% output：生成一个gif
% @Author     
% Copyright© 2022.5.22 CSDN name: cugautozp

        [x1,x2]=meshgrid(x1,x2);
        z=f(x1,x2);
        figure('color','w')
        sgtitle(['\it f=',f2s(f)])
        subplot(211) 
        mesh(x1,x2,z)
        axis off
        view([-35,45])
        hold on
        subplot(212)
        contour(x1,x2,z,20)
        zlim([0,0.5])
        set(gca,'ZTick',[],'zcolor','w')
        axis off
        view([-35,45])
        hold on

        pic_num = 1;

        for j =1:i-1    
            a=[x0{j}(1),x0{j}(2),f(x0{j}(1),x0{j}(2))];
            b=[x0{j+1}(1),x0{j+1}(2),f(x0{j+1}(1),x0{j+1}(2))];
            c=[a',b'];
            a1=[x0{j}(1),x0{j}(2)];
            b1=[x0{j+1}(1),x0{j+1}(2)];
            c1=[a1',b1'];
    
            subplot(211)
            plot3(x0{j}(1),x0{j}(2),f(x0{j}(1),x0{j}(2)),'r.','MarkerSize',10)
            subplot(212)
            plot(x0{j}(1),x0{j}(2),'r.','MarkerSize',10)
            drawnow
            F(j)=getframe(gcf);
            pause(0.5)

            subplot(211)
            plot3(c(1,:),c(2,:),c(3,:),'r--')    
            subplot(212)
            plot(c1(1,:),c1(2,:),'r--')
            drawnow 
            F(2*j)=getframe(gcf);
            pause(0.5)
            
            % 绘制并保存gif
            I=frame2im(F(j));
            [I,map]=rgb2ind(I,256);
            I1=frame2im(F(2*j));
            [I1,map1]=rgb2ind(I1,256);
            if pic_num == 1
                imwrite(I,map, GIFname ,'gif', 'Loopcount',inf,'DelayTime',0.5);
            else
                imwrite(I,map, GIFname ,'gif','WriteMode','append','DelayTime',0.5);
                imwrite(I1,map1, GIFname ,'gif','WriteMode','append','DelayTime',0.5);
            end
            pic_num = pic_num + 1;
        end
        subplot(211)
        plot3(x0{i}(1),x0{i}(2),f(x0{i}(1),x0{i}(2)),'r.','MarkerSize',9)
        subplot(212)
        plot(x0{i}(1),x0{i}(2),'r.','MarkerSize',9)
        F(2*i-1)=getframe(gcf);
        I=frame2im(F(2*i-1));
        [I,map]=rgb2ind(I,256);
        imwrite(I,map, GIFname ,'gif','WriteMode','append','DelayTime',0.5);
end