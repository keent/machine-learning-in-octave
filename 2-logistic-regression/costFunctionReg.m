function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;
hOfX = sigmoid(z);

Jfirst = (-1) * ((1./m) * sum (y .* log(hOfX) + ((1 - y) .* log(1 .- hOfX))));
theta_2 = theta;
theta_2(1, :) = [];
Reg = (lambda / (2 * m)) * (theta_2' * theta_2);
J = Jfirst + Reg;
    
iter = length(theta);

for n=1:iter,
    sumOfMTrainSet = (hOfX - y)' * X(:, n);
    if n == 1
        grad(n) = (1/m) * sumOfMTrainSet;
    else
        gReg = (lambda/m) * theta(n);
        grad(n) = ((1/m) * sumOfMTrainSet) + gReg;
    end
end



% =============================================================

end
