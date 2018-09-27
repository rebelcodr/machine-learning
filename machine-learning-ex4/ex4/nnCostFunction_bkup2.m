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


% Identity matrix to represent 0 to 9 numbers
iMatrix = eye(num_labels);

% Transform to get y matrix
yMatrix = zeros(m, num_labels);

% Loop over training samples and foreach output numeric number,
% and pick the right row vector that represents the number
for i=1:m
  yMatrix(i, :)= iMatrix(y(i), :);
end

% Add ones (bias unit) to the X data matrix
A1 = [ones(m, 1) X];

z2 = sigmoid(A1 * Theta1');

% Add ones (bias unit) to layer 2 units
m2 = size(z2, 1);
z2 = [ones(m2, 1) z2];
z3 = sigmoid(z2 * Theta2');

% Without regularization
% J = (1/m) * sum(sum(-yMatrix .* log(z3) - (1 - yMatrix) .* log(1-z3), 2));

% with regularization
Theta1Reg = sum(sum(Theta1(:, 2:end) .^ 2, 2));
Theta2Reg = sum(sum(Theta2(:, 2:end) .^ 2, 2));

J = (1/m) * sum(sum(-yMatrix .* log(z3) - (1 - yMatrix) .* log(1-z3), 2)) ...
+ ((lambda/(2*m)) * (Theta1Reg + Theta2Reg));

D = 0;
for t = 1:m
  % --- X(5000x400) => A1(401x1) ---
  A1 = [1; X(t, :)'];

  % --- A1 (401x1) Theta1 (25x401) => A2 (26x1) ---
  A2 = [1; sigmoid(Theta1 * A1)];

  % --- A2b (26x1) Theta2 (10x26) => A3(10x1)---
  A3 = sigmoid(Theta2 * A2);

  % --- yMatrix (1x10) A3 (10x1) => delta3(10x1)---
  delta3 = A3 - yMatrix(t, :)';

  % Compute hidden layer delta
  % --- Theta2 (10x26), delta3 (10x1) => delta2(26x1) ---
  delta2 = (Theta2' * delta3) .* sigmoidGradient(A2);
  % --- delta2(25x1) ---
  delta2 = delta2(2:end);

  % --- Theta2_grad(10x26), delta3(10x1), A2(26x1) => 10x26 ---
  Theta2_grad = Theta2_grad + delta3 * A2';

  % --- Theta1_grad(25x401), delta2(25x1), A1(401x1) => 25x401 ---
  Theta1_grad = Theta1_grad + delta2 * A1';
end

Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

% Add regularization
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1_grad(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2_grad(:, 2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
