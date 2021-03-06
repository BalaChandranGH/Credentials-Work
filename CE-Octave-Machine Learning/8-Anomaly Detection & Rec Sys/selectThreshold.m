function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;          % Scalar
bestF1 = 0;               % Scalar
F1 = 0;                   % Scalar 

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    % =============================================================
    
    % yval                                          % (m x 1) vector
    % pval                                          % (m x 1) vector
    
    Predictions_cv = (pval < epsilon);              % (m x 1) vector
    
    TP = sum((Predictions_cv == 1) & (yval == 1));  % (m x 1) vector
    FP = sum((Predictions_cv == 1) & (yval == 0));  % (m x 1) vector
    FN = sum((Predictions_cv == 0) & (yval == 1));  % (m x 1) vector
    
    Prec = TP / (TP + FP);                          % (m x 1) vector
    Rec  = TP / (TP + FN);                          % (m x 1) vector
    F1   = (2 * Prec * Rec) / (Prec + Rec); 
    
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
