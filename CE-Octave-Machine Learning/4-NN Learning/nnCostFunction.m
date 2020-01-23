function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------
% =========================================================================

% ========== PART 1: Compute Cost J without Regularization ==========

%%%%% Calculate the Unregularized cost %%%%%
% Add X0 to X, i.e., 1's as first column
  a1 = [ones(m, 1) X];
  
% Derive the hidden layer a2 
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);

% Add a0 to a2, i.e., 1's as the first column
  a2 = [ones(size(a2, 1), 1), a2];
  
% Derive the output layer/ hypothesis h(x)
  z3 = a2 * Theta2';
  hx = sigmoid(z3);

% Create y vector for K (=10) labels
  vec_y = (1:num_labels) == y;
  
% Compute the Unregularized cost using element-wise multiplication...
% for all K lables, then for the entire training set and get a scalar
  J = (1 / m) * sum(sum((-vec_y .* log(hx)) - ((1 - vec_y) .* log(1 - hx))));

  
% ========== PART 2: Implement Backprobagation algorithm ==========
  D3 = hx - vec_y;
  D2 = (D3 * Theta2) .* [ones(size(z2, 1), 1) sigmoidGradient(z2)];
  D2 = D2(:, 2:end);
  
  Theta1_grad = (1 / m) * (D2' * a1);
  Theta2_grad = (1 / m) * (D3' * a2);

  
% ========== PART 3: Compute Cost J  & grad with Regularization ==========
%%%%% Calculate the Regularized Cost %%%%%
% Calculate the Regularization Cost term
  RCost_Term = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^2)) + sum(sum(Theta2(:, 2:end) .^2)));

  % Regulasize the Cost function J
  J = J + RCost_Term;

% Calculate the Regularization grad term for Theta1_grad & Theta2_grad
  Rgrad_Term_Theta1 = (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
  Rgrad_Term_Theta2 = (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
  
% Regulasize the gradients
  Theta1_grad = Theta1_grad + Rgrad_Term_Theta1;
  Theta2_grad = Theta2_grad + Rgrad_Term_Theta2;
  
  
% Unroll gradients
  grad = [Theta1_grad(:); Theta2_grad(:)];

end
