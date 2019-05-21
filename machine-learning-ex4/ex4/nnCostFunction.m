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
X=[ones(size(X,1),1) X];
hhh=(X*Theta1');
hh=sigmoid(hhh);
hh=[ones(size(hh,1),1) hh];
h=hh*Theta2';
h=sigmoid(h);
size(h);
yy=y;
yy(yy==0)=10;
y2=[0:size(Theta2,1):(size(X,1)*size(Theta2,1))-1];
yy=yy+y2';
size(y2);
y1=zeros(size(X,1),size(Theta2,1));
size(y1);
sum(sum(y1));
y1=y1';
y1(yy)=1;
y1=y1';
sum(sum(y1));
a1=-(y1.*log(h))-((1-y1).*log(1-h));
J=(1/m).*sum(sum(a1));
theta1=Theta1(:,2:end).^2;
theta2=Theta2(:,2:end).^2;
s=sum(sum(theta1))+sum(sum(theta2));
J=J+((lambda*s)/(2*m));

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
size(Theta2_grad);
a2=h-y1;
size(a2);
size(hh);
Theta2_grad = a2' * hh;
Theta2_grad=Theta2_grad/m;
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+((lambda/m)*Theta2(:,2:end));
size(Theta2_grad);

a3=(a2*Theta2).*sigmoidGradient([ones(size(hhh,1),1) hhh]);
a3=a3(:,2:end);
size(a3);
size(Theta1_grad);
Theta1_grad=a3'*X;
Theta1_grad=Theta1_grad/m;
Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+((lambda/m)*Theta1(:,2:end));
size(Theta1_grad);
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
