%% Calibration of the first camera
% The given image coordinates were originally localized manually using
im1=imread('im1.jpg');

x1 = 1.0e+03 * [0.7435; 3.3315; 0.8275; 3.2835;
                0.5475; 3.9875; 0.6715; 3.8835];
            
y1 = 1.0e+03 * [0.4455; 0.4335; 1.7215; 1.5615;
                0.3895; 0.3895; 2.1415; 1.8735];

% Image coordinates of points as rows of matrix 'abcdefgh'
abcdefgh=[x1(:) y1(:)];

% World coordinates of the points (dimensions of the shelf)
ABCDEFGH=[758  0 -295;
          0  0 -295;
          758  360 -295;
          0  360 -295;
          758  0 0;
          0  0 0;
          758  360 0;
          0  360 0];

% Plot manualy localized points  
figure;imshow(im1);hold on
title('Cyan: manually located points  Red: projected points')
plot(x1,y1,'c+','MarkerSize',10);
labels={'a','b','c','d','e','f','g','h'};
for i=1:length(x1)
    ti=text(x1(i),y1(i),labels{i});
    ti.Color='cyan';
    ti.FontSize=20;
end


%% Your task is to implement the missing function 'camcalibDLT.m'.
% The algorithm is summarised on the lecture slides and exercise sheet.
% The function takes the homogeneous coordinates of the points as input.
P1 = camcalibDLT([ABCDEFGH ones(8,1)], [abcdefgh ones(8,1)]);

% Check the results by projecting the world points with the estimated P
% the projected points should overlap with manually localized points
pproj1 = P1*[ABCDEFGH';ones(1,8)];
for i=1:8
    plot(pproj1(1,i)/pproj1(3,i),pproj1(2,i)/pproj1(3,i),'rx','MarkerSize',20);
end


%% Calibration of the second camera
im2 = imread('im2.jpg');

x2 = 1.0e+03 * [0.5835; 3.2515; 0.6515; 3.1995;
               0.1275; 3.7475; 0.2475; 3.6635];
           
y2 = 1.0e+03 * [0.4135; 0.4015; 1.6655; 1.5975;
                0.3215; 0.3135; 2.0295; 1.9335];

figure;imshow(im2);hold on
title('Cyan: manually located points  Red: projected points')
plot(x2,y2,'c+','MarkerSize',10);
labels={'a','b','c','d','e','f','g','h'};
for i=1:length(x2)
    ti=text(x2(i),y2(i),labels{i});
    ti.Color='cyan';
    ti.FontSize=20;
end
%% Second camera projection matrix
P2 = camcalibDLT([ABCDEFGH ones(8,1)], [x2 y2 ones(8,1)]);

% Again, check results by projecting world coordinates with P2
pproj2 = P2*[ABCDEFGH';ones(1,8)];
for i=1:8 
    plot(pproj2(1,i)/pproj2(3,i),pproj2(2,i)/pproj2(3,i),'rx','MarkerSize',20);
end

