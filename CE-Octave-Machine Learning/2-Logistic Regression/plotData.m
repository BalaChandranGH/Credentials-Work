function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%









% =========================================================================
% Find indices for Positive and Negative examples
  P = find(y == 1); N = find(y == 0);

% Plot a Scatter Plot for the examples (Training data)
  plot(X(P, 1), X(P, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
  plot(X(N, 1), X(N, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
  hold off;

end
