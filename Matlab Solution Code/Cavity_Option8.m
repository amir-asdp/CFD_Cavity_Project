clc;
clear;
close all;
format shortEng

%% Defining Parameters
U = 1;
L = 1;
H = 1;
Re = 100;
nu = U * L / Re;
nx = 101;
ny = 101;
delta_x = L / (nx - 1);
delta_y = H / (ny - 1);
beta =delta_x / delta_y;
beta2 = beta.^2;

%% Define Matrices of Stream Function & Vorticity
sf = zeros(nx,ny);
sf0= zeros(nx,ny);
w = zeros(nx, ny);
w0 = zeros(nx, ny);
u = zeros(nx, ny);  u(:,1) = -U;  u(:,end) = U;  %Top moves right & Bottom moves left
v = zeros(nx, ny);

%% Findng Stream Function & Velocity & Vorticity & Error
Err_tot = 100;
step = 0;

while ( Err_tot > 1e-3)
    
    %Finding Stream Function
    for i=2:nx-1
        for j=2:ny-1
            sf(i,j) = (sf(i+1,j) + sf(i-1,j) +...
                       beta2* sf(i,j+1) + beta2* sf(i,j-1) + ...
                       (delta_x.^2) * w(i,j)) / (2 + 2* beta2);
        end
    end
    
    
    %Update Boundary Condition for Vorticity
    w(1,:)=    -2 * sf(2,:)/delta_x^2;
    w(end,:)=  -2 * sf(end-1,:)/delta_x^2; 
    w(:,1)=    -2 * (sf(:,2) + U*delta_y)/delta_y^2;
    w(:,end)=  -2 * (sf(:,end-1) + U*delta_y)/delta_y^2;

    
    %Finding Velocity & Vorticity
    for i=2:nx-1
        for j=2:ny-1
            u(i,j) = (sf(i,j+1) - sf(i,j-1)) / (2 * delta_y);
            v(i,j) = -(sf(i+1,j) - sf(i-1,j)) / (2 * delta_x);

            dwdx = (w(i+1,j) - w(i-1,j)) / (2 * delta_x);
            dwdy = (w(i,j+1) - w(i,j-1)) / (2 * delta_y);

            A = ((u(i,j) * dwdx) + (v(i,j) * dwdy))/ nu;

            w(i,j) = (w(i+1,j) + w(i-1,j) + beta2* w(i,j+1) + beta2* w(i,j-1) - (delta_x.^2) * A) / (2 + 2* beta2);
        end
    end
    V_tot = sqrt(u.^2 + v.^2);

    
    %Finding Error
    Err_sf = max(max(abs(sf0-sf)));
    Err_w = max(max(abs(w0-w)));
    Err_tot = max([Err_sf , Err_w]);
    disp(Err_tot);
    
    
    %Update sf0 & w0
    w0 = w;
    sf0 = sf;
    step = step + 1;
    
    
    %Extract GIF Of Contours
    if (mod(step,50)==0)
        
        %GIF For u
        ugif=figure(1);
        contourf(transpose(u),50);
        title('Contour of velocity u');
        xlabel('x(m)'); 
        ylabel('y(m)'); 
        colorbar
        colormap jet
        drawnow
        frame1 = getframe(ugif);
        im1 = frame2im(frame1);
        [imind1,cm1] = rgb2ind(im1,256);
        if step == 50
            imwrite(imind1,cm1,'Cavity_Option8_u.gif','gif', 'Loopcount',inf);
        else
            imwrite(imind1,cm1,'Cavity_Option8_u.gif','gif','WriteMode','append');
        end
        
        %GIF For v
        vgif=figure(2);
        contourf(transpose(v),50);
        title('Contour of velocity v');
        xlabel('x(m)'); 
        ylabel('y(m)'); 
        colorbar
        colormap jet
        drawnow
        frame2 = getframe(vgif);
        im2 = frame2im(frame2);
        [imind2,cm2] = rgb2ind(im2,256);
        if step == 50
            imwrite(imind2,cm2,'Cavity_Option8_v.gif','gif', 'Loopcount',inf);
        else
            imwrite(imind2,cm2,'Cavity_Option8_v.gif','gif','WriteMode','append');
        end
        
        %GIF For total Velocity
        vtgif=figure(3);
        contourf(transpose(V_tot));
        title('Contour of total velocity');
        xlabel('x(m)'); 
        ylabel('y(m)'); 
        colorbar
        drawnow
        frame3 = getframe(vtgif);
        im3 = frame2im(frame3);
        [imind3,cm3] = rgb2ind(im3,256);
        if step == 50
            imwrite(imind3,cm3,'Cavity_Option8_V-total.gif','gif', 'Loopcount',inf);
        else
            imwrite(imind3,cm3,'Cavity_Option8_V-total.gif','gif','WriteMode','append');
        end
        
        %GIF For Stream Line
        sfgif=figure(4);
        contourf(transpose(sf),50);
        title('Contour of Stream Lines');
        xlabel('x(m)'); 
        ylabel('y(m)'); 
        colorbar
        colormap jet
        drawnow
        frame4 = getframe(sfgif);
        im4 = frame2im(frame4);
        [imind4,cm4] = rgb2ind(im4,256);
        if step == 50
            imwrite(imind4,cm4,'Cavity_Option8_Stream Lines.gif','gif', 'Loopcount',inf);
        else
            imwrite(imind4,cm4,'Cavity_Option8_Stream Lines.gif','gif','WriteMode','append');
        end
        
        %GIF For Vorticity
        wgif=figure(5);
        contourf(transpose(w),1000);
        title('Contour of Vorticity');
        xlabel('x(m)');
        ylabel('y(m)');
        colorbar
        colormap jet
        drawnow
        frame5 = getframe(wgif);
        im5 = frame2im(frame5);
        [imind5,cm5] = rgb2ind(im5,256);
        if step == 50
            imwrite(imind5,cm5,'Cavity_Option8_Vorticity.gif','gif', 'Loopcount',inf);
        else
            imwrite(imind5,cm5,'Cavity_Option8_Vorticity.gif','gif','WriteMode','append');
        end
    end

end


figure(6);
hold on
contour(transpose(sf),10);
q = quiver(transpose(u),transpose(v),3);
q.Color = 'r';
xlabel('x(m)');
ylabel('y(m)');
title('Stream Line & Vector of Velocity');
legend('Stream Line', 'Vector of velocity')
hold off




