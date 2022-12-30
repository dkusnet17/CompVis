function [ F ] = estimateFnorm(x1,x2)
    % Inputs:
    %   x1: Points from image 1, with shape (coordinates, point_id)
    %   x2: Points from image 2, with shape (coordinates, point_id)
    % Outputs:
    %   F: Estimated fundamental matrix
    
    % Normalise each set of points so that the origin 
    % is at centroid and mean distance from origin is sqrt(2). 
    % normalise2dpts also ensures the scale parameter is 1.
    [x1, T1] = normalise2dpts(x1);
    [x2, T2] = normalise2dpts(x2);
    
    % Use eight-point algorithm to estimate fundamental matrix F
    F = estimateF(x1, x2);
    
    % Denormalise back to original coordinates
    %%-your-code-starts-here-%%

    %%-your-code-ends-here-%%
    
end

