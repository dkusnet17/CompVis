%% Setup
% Create the webcam object, or use a video file.
use_webcam = true;  % set to false if you want use a video file
if use_webcam
    cam = webcam();
else
    cam = VideoReader('visionface.avi');  % video prodived as part of the Vision toolbox
    video_start = cam.CurrentTime;
end


% Create the video player object.
sz = [480 640];  % frame size
videoPlayer = vision.VideoPlayer('Position', [100 100 sz(2)+50 sz(1)+50]);

% Cascade detector instance
face_cascade = vision.CascadeObjectDetector('haarcascade_frontalface_default.xml');

% Parametersface_cascade.ScaleFactor = 1.2;  % image pyramid scale
face_cascade.MergeThreshold = 5;  

% Point tracker object.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);


%% USE MATLAB DOCUMENTATION TO FIND OUT HOW CERTAIN FUNCTIONS WORK.

% Loop parameters
runLoop = true;
p0 = [];

while runLoop

    % Get a single frame.
    if use_webcam
        img = snapshot(cam);
        img = imresize(img, sz);
    else
        if ~cam.hasFrame
            cam.CurrentTime = video_start;
        end
        img = cam.readFrame();
    end

    % Mirror
    img = fliplr(img);
    
    % Grayscale copy
    img_gray = rgb2gray(img);
    
    if size(p0, 1) <= 3
        % Detection
        img = insertText(img, [0,0], 'Detection');
        
        % Detect faces. Detections are in form 
        % (x_upperleft, y_upperleft, width, height)
        faces = face_cascade.step(img_gray);
    
        % Take the first face and get trackable points.
        if ~isempty(faces)
            % Extract ROI (face) from the grayscale frame
            % You can also crop this ROI even more to make sure only 
            % the face area is considered in the tracking.
            
            %%-your-code-starts-here-%%
            roi_gray = zeros(size(img_gray));  % replace me
            %%-your-code-starts-here-%%
            
            % Find corner points inside the detected region.
            points = detectMinEigenFeatures(roi_gray);
            p0 = points.Location;  % xy coordinates
            
            % Convert from ROI to image coordinates
            %%-your-code-starts-here-%%

            %%-your-code-ends-here-%%
            
            % Initialized point tracker
            if ~isempty(p0)
                release(pointTracker);
                initialize(pointTracker, p0, img_gray);      
            end
        end
    else
        % Tracking
        img = insertText(img, [0,0], 'Tracking'); 

        % Calculate optical flow using pointTracker
        [p1, isFound] = step(pointTracker, img_gray);
        
        % Select good points. First return value from our tracker
        % is the new points, second return value is the points which were
        % found from previous frame, and the third argument indicates if 
        % an error occurred. Use the second return value save only the points
        % which were found from previous frame.         
        %%-your-code-starts-here-%%

        %%-your-code-ends-here-%%
        
        % Draw points using insertMarker
        %%-your-code-starts-here-%%

        %%-your-code-ends-here-%%

        % Update p0 for next iteration (which points should be kept?)
        %%-your-code-starts-here-%%

        %%-your-code-ends-here-%%
        
        % Update pointTracker if p0 is found
        if ~isempty(p0)
            setPoints(pointTracker, p0);
        end
        
    end
 
    % Display the annotated video frame using the video player object.
    step(videoPlayer, img);
    if ~use_webcam
       pause(0.02) 
    end
    
    % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);
    
end

% Release camera
clear cam

