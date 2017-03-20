function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


Theta_temp = zeros(length(theta), 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    
    h = X * theta;
    difference = h - y;

    num_itr = length(theta)
    for itr = 1:num_itr        
        if (itr == 1)
            sum = ( (difference') * (ones((length(X)), 1)) );
        else
            sum = ( (difference') * (X(:, itr)) );
        endif
        Theta_temp(itr) = theta(itr) - (alpha * (1/m) * sum);
    end

    theta = Theta_temp; 








    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
