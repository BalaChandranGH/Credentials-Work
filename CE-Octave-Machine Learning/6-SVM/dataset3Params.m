function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% =========================================================================

  % X          (m x n) matrix, 51x2 for ex6data1.mat, 863x2 for ex6data2.mat, 211x2 for ex6data3.mat
  % y          (m x 1) matrix, 51x1 for ex6data1.mat, 863x1 for ex6data2.mat, 211x1 for ex6data3.mat
  % Xval       (m x n) matrix, 200x2 from ex6data3.mat
  % yval       (m x 1) matrix, 200x1 from ex6data3.mat
  % C          (1 x 1) Scalar, Return value - optimal learning parameter
  % sigma      (1 x 1) Scalar, Return value - optimal learning parameter
  
  C_list     = [0.01 0.03 0.1 0.3 1 3 10 30];  % List of values for C
  sigma_list = [0.01 0.03 0.1 0.3 1 3 10 30];  % List of values for sigma
  
  Prediction_Err = zeros(length(C_list), length(sigma_list));   % (8 x 8) matrix
  
  % Train and Evaluate 64 models (8 x 8 = 64, i.e, # of C by # of sigma values)
  for i = 1:length(C_list)
    for j = 1:length(sigma_list)
      C_val = C_list(i);
      sigma_val = sigma_list(j);
      model = svmTrain(X, y, C_val, @(x1, x2) gaussianKernel(x1, x2, sigma_val));
      Predictions = svmPredict(model, Xval);
      Prediction_Err(i, j) = mean(double(Predictions ~= yval));
    endfor
  endfor
  
  % Find out the lowest prediction error w.r.t. Row and Column
  [values row_indexes] = min(Prediction_Err);
  [~, col] = min(values);
  row = row_indexes(col);
  
  C = C_list(row);          
  sigma = sigma_list(col);  
 
end
