function P=camcalibDLT(x_world, x_im)
    % Inputs: 
    %   x_world: World coordinates with shape (point_id, coordinates)
    %   x_im: Image coordinates with shape (point_id, coordinates)
    %
    % Outputs:
    %   P: Camera projection matrix with shape (3,4)
    
    % Create the matrix A 
    %%-your-code-starts-here-%%
    n = 8;
    A = zeros(16,12);
    y = x_im(:,2);
    x = x_im(:,1);
    for i=1:n 
        A(i*2-1,:) = [zeros(1,4) x_world(i,:) -y(i)*x_world(i,:)];
        A(i*2,:) = [x_world(i,:) zeros(1,4) -x(i)*x_world(i,:)];
    end
    disp(A)
    %%-your-code-ends-here-%%
    
    % Perform homogeneous least squares fitting.
    % The best solution is given by the eigenvector of 
    % A.T*A with the smallest eigenvalue.
    %%-your-code-starts-here-%%
    eigvs=A'*A;
    [V,D] = eig(eigvs)
    [~,ind]= min(diag(D));
    ev= V(:,ind);
    %%-your-code-ends-here-%%

    % Reshape the eigenvector into a projection matrix P
    P = (reshape(ev,4,3))'; % here ev is the eigenvector above
    %P = [0 0 0 0;0 0 0 0;0 0 0 1];  % remove this and uncomment the above
end