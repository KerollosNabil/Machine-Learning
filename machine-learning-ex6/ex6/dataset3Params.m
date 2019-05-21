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

x1 = [1 2 1]; x2 = [0 4 -1];
min =100000000000000;
a=[0.01, 0.03, 0.1, 0.3, 0.5, 1, 3, 10, 30];
for z = 1:9
    for zz = 1:9
        Cc=a(z);
        sigmaa = a(zz);
        modell= svmTrain(X, y, Cc, @(x1, x2) gaussianKernel(x1, x2, sigmaa));
        prediction = svmPredict(modell,Xval);
        
        if mean(double(prediction ~= yval))< min
            min = mean(double(prediction ~= yval));
            C=a(z);
            sigma = a(zz);
        end
        
        
    end
end

C
sigma

% =========================================================================

end
