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






% =============================================================
% ===== Compute Regularized Cost =====
  % Evaluate the hypothesis
    h = sigmoid(X * theta);
    
  % Calculate the Unregularized cost
    UR_Cost = (1 / m) * (((-1 * y)' * log(h)) - ((1 - y)' * log(1 - h)));
  
  % Let Regularization exclude the Bias feature by setting theta(1) = 0
    theta(1) = 0;
  
  % Calculate the Regularization Cost term
    RCost_Term = (lambda / (2 * m)) * (theta' * theta);
  
  % Calculate the Cost
    J = UR_Cost + RCost_Term;
  
% ===== Compute Gradient =====
  % Calculate the Regularization Gradient Term
    RGrad_Term = (lambda / m) * theta;
  
% Calculate the Gradient  
    grad = ((1 / m) * (X' * (h - y))) + RGrad_Term;
  
end
