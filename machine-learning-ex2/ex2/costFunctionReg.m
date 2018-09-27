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

h = sigmoid(X * theta);
n = size(theta);

J = (-y' * log(h) - (1 - y)' * log(1 - h)) * (1/m) + ((lambda/(2*m)) * sum(theta(2:n) .^ 2));


% grad_0 = (X(:, 1)' * (sigmoid(X(:, 1) * theta(1, 1)) - y)) * (1/m);
% grad1_n = ((X(:, 2:size(theta))' * (sigmoid(X(:, 2:n) * theta(2:n, 1)) - y)) * (1/m)) + (lambda/m * theta(2:n));
% grad = [grad_0; grad1_n]

grad = (X' * (h - y)) * (1/m) + ((lambda/m) * [0; theta(2:n)]);


% =============================================================

end
