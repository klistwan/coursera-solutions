function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

for i = 1: m
  temporary(i) = 0;
  for j = 1: size(theta)
    temporary(i) = temporary(i) + theta(j) * X(i, j);
  end
end
    
for i = 1: m
  h(i) = sigmoid(temporary(i));
end

for i = 1: m
  J = J + y(i) * log(h(i)) + (1-y(i)) * log(1-h(i));
end

J = -J / m;

for i = 1 : size(theta)
  grad(i) = 0;
  for j = 1 : m
    grad(i) = grad(i) + (h(j) - y(j)) * X(j,i);
  end
  grad(i) = grad(i) / m;
end

% Cost of H(x) = -y log(H(x) - (1-y) * log (1 - H(x))

% Code I did for cost function of linear regression. 
% predictions = X*theta;
% sqrErrors = (predictions-y) .^2;
% J = 1/(2*m) * sum(sqrErrors);






% =============================================================

end
