function [ F ] = estimateF(x1, x2)
    % Inputs:
    %   x1: Points from image 1, with shape (coordinates, point_id)
    %   x2: Points from image 2, with shape (coordinates, point_id)
    % Outputs:
    %   F: Estimated fundamental matrix

    % Use x1 and x2 to construct equation for homogeneous linear system
    %%-your-code-starts-here-%%

    %%-your-code-ends-here-%%
   
    % Use SVD to find the solution for this homogeneous linear system by
    % extracting the row from V corresponding to the smallest singular value.
    %%-your-code-starts-here-%%

    %%-your-code-ends-here-%%
    %F = reshape(v_min,3,3)';  % reshape to acquire Fundamental matrix F
    F = ones(3);  % remove me and uncomment the above
    
    % Enforce constraint that fundamental matrix has rank 2 by performing
    % SVD and then reconstructing with only the two largest singular values.
    % Reconstruction is done with u * s * vh where s is the singular values
    % in a diagonal form.
    %%-your-code-starts-here-%%

    %%-your-code-ends-here-%%
    
end

