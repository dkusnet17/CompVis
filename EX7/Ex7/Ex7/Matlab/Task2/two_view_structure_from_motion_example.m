%% Load the images and their camera matrices (two views of the same scene)
im1 = imread('im1.jpg');
im2 = imread('im2.jpg');
%% Load the camera matrices P1 and P2 from 'P1.mat' and 'P2.mat' and the image corner coordinates from 'cornercoordinates.mat'.

%%-your-code-starts-here-%%

%%-your-code-ends-here-%%

%% Corner projection and 3D sketch
% Give labels for the corners of the shelf
labels = {'a','b','c','d','e','f','g','h'};

% Define the 3D coordinates of the corners based on the known dimensions 
ABCDEFGH = [758  0 -295;
            0  0 -295;
            758  360 -295;
            0  360 -295;
            758  0 0;
            0  0 0;
            758  360 0;
            0  360 0];
      
% Visualize the corners in the images      
figimgs=figure;hold on
subplot(2, 1, 1); imshow(im1); hold on
plot(x1,y1,'c+','MarkerSize',10);
subplot(2, 1, 2); imshow(im2); hold on
plot(x2,y2,'c+','MarkerSize',10);
for i=1:length(x1)
    subplot(2, 1, 1)
    ti=text(x1(i),y1(i),labels{i});
    ti.Color='cyan';
    ti.FontSize=20;
    subplot(2, 1, 2)
    ti=text(x2(i),y2(i),labels{i});
    ti.Color='cyan';
    ti.FontSize=20;
end

% Calibrate the cameras from 3D <-> 2D correspondences
P1t = camcalibDLT([ABCDEFGH ones(8,1)], [x1 y1 ones(8,1)]);
P2t = camcalibDLT([ABCDEFGH ones(8,1)], [x2 y2 ones(8,1)]);

% Visualize a 3D sketch of the shelf
edges=[1 2;1 3;3 4; 2 4; 1 5; 5 6; 2 6; 5 7; 3 7;4 8;7 8;6 8]';
figure;hold on;
title('3D sketch of the shelf')
for i=1:size(edges,2)
    plot3(ABCDEFGH(edges(:,i),1),ABCDEFGH(edges(:,i),2),ABCDEFGH(edges(:,i),3),'k-');
end
for i=1:8
    ti=text(ABCDEFGH(i,1),ABCDEFGH(i,2),ABCDEFGH(i,3),labels{i});
    ti.FontSize=20;
end
axis equal;
view([1 1 1]);


%% Your task is to project the 3D corners in ABCDEFGH (currently non-homogeneous) to
% images 1 and 2 using P1t and P2t, i.e. the calibrated camera matrices.
% Convert the projected points back to non-homogeneous form and store them into the given variables below.
% Results are visualized using magenta lines.
%%-your-code-starts-here-%%
cx1 = zeros(size(edges,2), 1);  % replace me
cy1 = zeros(size(edges,2), 1);  % replace me
cx2 = zeros(size(edges,2), 1);  % replace me
cy2 = zeros(size(edges,2), 1);  % replace me
%%-your-code-stops-here-%%

% Illustrate the edges of the shelf that connect its corners 
% cx1 and cy1 are x and y coordinates for image 1, and similarly for image 2
figure(figimgs);
for i=1:size(edges,2)
     subplot(2, 1, 1)
     plot(cx1(edges(:,i)),cy1(edges(:,i)),'m-');
     subplot(2, 1, 2)
     plot(cx2(edges(:,i)),cy2(edges(:,i)),'m-');
end

%% Compute a projective reconstruction of the shelf
% That is, triangulate the corner correspondences using the camera projection 
% matrices which were recovered from the fundamental matrix.
Ps{1}=P1;Ps{2}=P2;
Xcorners=zeros(4,8);

P = Ps;
imsize = [size(im1,2) size(im2,2);size(im1,1) size(im2,1)];

for i=1:8
   % The following function is from http://www.robots.ox.ac.uk/~vgg/hzbook/code/
   Xcorners(:,i)=vgg_X_from_xP_lin([x1(i) x2(i);y1(i) y2(i)],Ps,[size(im1,2) size(im2,2);size(im1,1) size(im2,1)]);
end
Xc=Xcorners(1:3,:)./Xcorners([4 4 4],:);

% Visualize the projective reconstruction
% notice that the shape is not a rectangular cuboid 
% (there is a projective distortion)
figure;hold on;
title('Projection reconstruction (try rotating the shape)')
for i=1:size(edges,2)
    plot3(Xc(1,edges(:,i))',Xc(2,edges(:,i))',Xc(3,edges(:,i))','k-');
end
for i=1:8
    ti=text(Xc(1,i),Xc(2,i),Xc(3,i),labels{i});
    ti.FontSize=20;
end

%% Your next task is to project the cuboid corners 'Xc' (currently non-homogeneous) to images 1 and 2.
% Use camera projection matrices P1 and P2 (recovered from the fundamental matrix).
% Results are visualized using cyan lines.
% The cyan edges should be relatively close to the magenta lines from above.
% Convert the projected points back to non-homogeneous form and store into the given variables below.

%%-your-code-starts-here-%%
pcx1 = zeros(size(edges,2), 1);  % replace me
pcy1 = zeros(size(edges,2), 1);  % replace me
pcx2 = zeros(size(edges,2), 1);  % replace me
pcy2 = zeros(size(edges,2), 1);  % replace me
%%-your-code-starts-here-%%

% pcx1 and pcy1 are x and y coordinates for image 1, and similarly for image 2
figure(figimgs);
title('Cyan: projected cuboid')
for i=1:size(edges,2)
    subplot(2, 1, 1)
    plot(pcx1(edges(:,i)),pcy1(edges(:,i)),'c-');
    subplot(2, 1, 2)
    plot(pcx2(edges(:,i)),pcy2(edges(:,i)),'c-');
end


