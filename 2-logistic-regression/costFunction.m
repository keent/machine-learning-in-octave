function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

alpha = 0.01;
z = X * theta;
hOfX = sigmoid(z);

J = (-1) * ((1./m) * sum (y .* log(hOfX) + ((1 - y) .* log(1 .- hOfX))));

iter = length(theta);

for n=1:iter,
    sumOfMTrainSet = (hOfX - y)' * X(:, n);
    %grad(n) = theta(n) - (alpha * (1/m) * sumOfMTrainSet);
    grad(n) = (1/m) * sumOfMTrainSet;
end


% =============================================================

end
