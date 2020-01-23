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
% m      -> (1 x 1) scalar
% X      -> (m x n) matrix (No. of samples x No. of features)
% y      -> (m x 1) vector
% lambda -> (1 x 1) scalar
% theta  -> (n x 1) vector
% grad   -> (n x 1) vector 
%
% =========================================================================

% Compute Regularized Cost
  h = (X * theta);                                                 % (m x 1) vector
  RegTerm_Cost = (lambda / (2*m)) * sum(theta(2:end) .^2);         % (1 x 1) scalar
  J = ((1/(2*m)) * sum((h - y) .^2)) + RegTerm_Cost;               % (1 x 1) scalar (Regularized Cost)
  
% Compute Regularized Gradient
  RegTerm_Grad = (lambda / m) * theta(2:end);                      % (n x 1) vector
  grad(1)     =  (1/m) * (X(:, 1)' * (h - y));                     % (1 x 1) scalar
  grad(2:end) = ((1/m) * (X(:, 2:end)' * (h - y))) + RegTerm_Grad; % (n x 1) vector

% Transpose grad from 'column vector' to 'row vector'
  grad = grad(:);                                                  % (1 x n) vector (Regularized Gradient)

end
