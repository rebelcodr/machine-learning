function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for i = 1:K
  % Get an identity matrix of size m
  xIMatrix = eye(m);

  % Get matching indices of current centroid
  indices = find(idx == i);

  % Get a row of size m, i.e. for the current centroid index,
  % get all X indices that belong to it. is: (1 x m)
  xIndex = sum(xIMatrix(indices, :));

  % Get count of non-zeroes
  total = numel(indices);

  % Multiply with X to get the sum of current centroid
  % belonging X values. Size is: (1 x m) * (m x n) => (1 x n)
  if total != 0
    centroids(i, :) = (1/total) * (xIndex * X);
  endif

end

% =============================================================
% Much easier way using logical arrays

% for k=1:K
    % use logical arrays for indexing
    % see http://www.mathworks.com/help/matlab/math/matrix-indexing.html#bq7egb6-1
  %  indexes = idx == k;
  %  centroids(k, :) = mean(X(indexes, :));
% end;



% =============================================================


end
