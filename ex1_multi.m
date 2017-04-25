%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization


%% Clear and Close Figures
clear all; close all; clc

fprintf('Loading data ...\n');
 
%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 1: Feature Normalization ================
	%%%% ======== Complete featureNormalize.m ===========
	function [X_norm, mu, sigma] = featureNormalize(X)
	%FEATURENORMALIZE Normalizes the features in X 
	%   FEATURENORMALIZE(X) returns a normalized version of X where
	%   the mean value of each feature is 0 and the standard deviation
	%   is 1. This is often a good preprocessing step to do when
	%   working with learning algorithms.
	
	% You need to set these values correctly
	X_norm = X;
	mu = zeros(1, size(X, 2));
	sigma = zeros(1, size(X, 2));
	
	% =========== YOUR CODE HERE ==============
	% Instructions: First, for each feature dimension, compute the mean
	%               of the feature and subtract it from the dataset,
	%               storing the mean value in mu. Next, compute the 
	%               standard deviation of each feature and divide
	%               each feature by it's standard deviation, storing
	%               the standard deviation in sigma. 
	%
	%               Note that X is a matrix where each column is a 
	%               feature and each row is an example. You need 
	%               to perform the normalization separately for 
	%               each feature. 
	%
	% Hint: You might find the 'mean' and 'std' functions useful.
	%       
	m = size(X , 1);
	mu = mean(X);
	for i = 1 : m,
		X_norm(i, :) = X(i , :) - mu;
	end
	
	sigma = std(X);
	for i = 1 : m,
		X_norm(i, :) = X_norm(i, :) ./ sigma;
	end
	
	%mu , sigma , X_norm
	
	% ========================================
	end
	%%%% ======== Complete featureNormalize.m ===========


% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================
	%%%% ======== Complete computeCostMulti.m ===========
	function J = computeCostMulti(X, y, theta)
	%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
	%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
	%   parameter for linear regression to fit the data points in X and y
	
	% Initialize some useful values
	m = length(y); % number of training examples
	
	% You need to return the following variables correctly 
	J = 0;
	
	% ======= YOUR CODE HERE =========
	% Instructions: Compute the cost of a particular choice of theta
	%               You should set J to the cost.
	
	J = sum((X * theta - y).^2) / (2 * m);
	% ================================
	end
	%%%% ======== Complete computeCostMulti.m ===========

	%%%% ======== Complete gradientDescentMulti.m ===========
	function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
	%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
	%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
	%   taking num_iters gradient steps with learning rate alpha
	
	% Initialize some useful values
	m = length(y); % number of training examples
	n = size(X , 2);
	J_history = zeros(num_iters, 1);
	
	for iter = 1:num_iters
	
	    % ============ YOUR CODE HERE ============
	    % Instructions: Perform a single gradient step on the parameter vector
	    %               theta. 
	    %
	    % Hint: While debugging, it can be useful to print out the values
	    %       of the cost function (computeCostMulti) and gradient here.
	    %
	
		H = X * theta;
		T = zeros(n , 1);
		for i = 1 : m,
			T = T + (H(i) - y(i)) * X(i,:)';	
		end
		
		theta = theta - (alpha * T) / m;
	
	    % ========================================
	
	    % Save the cost J in every iteration    
	    J_history(iter) = computeCostMulti(X, y, theta);
	
	end
	
	end
	%%%% ======== Complete gradientDescentMulti.m ===========

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.3;
num_iters = 100;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = 0; % You should change this


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================
	%%%% ======== Complete normalEqn.m ===========
	function [theta] = normalEqn(X, y)
	%NORMALEQN Computes the closed-form solution to linear regression 
	%   NORMALEQN(X,y) computes the closed-form solution to linear 
	%   regression using the normal equations.
	
	theta = zeros(size(X, 2), 1);
	
	% ========= YOUR CODE HERE ============
	% Instructions: Complete the code to compute the closed form solution
	%               to linear regression and put the result in theta.
	%	
	
	% ------ Sample Solution ------
	
	theta = pinv(X' * X) * X' * y;
	
	% =====================================
	end
	%%%% ======== Complete normalEqn.m ===========

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2-Copy1.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = 0; % You should change this


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

