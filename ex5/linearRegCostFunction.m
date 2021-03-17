function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J = sumsq( X * theta - y ) / (2 * m);

% regularization
J += lambda / (2 * m) * sumsq(theta(2:end));

%Gradient
for i = 1:size(theta)(1)
  grad(i) = X'(i,:)*(X * theta - y); % vector where each position represent derivative to respective theta param
endfor

grad /= m; 
grad(2:end) += (theta(2:end) * (lambda / m)); 

% =========================================================================

grad = grad(:);

end
