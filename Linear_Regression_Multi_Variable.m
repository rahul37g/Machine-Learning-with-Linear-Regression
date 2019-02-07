% Linear Regression with multiple variables

%% ============================================
%% ======= Part 1: Feature Normalization ======
%% ============================================= 

% Clear and Close figures
clear ; close all; clc

fprintf('Loading Data ... \n');

%% Load Data
data = load('dataset.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points from the entire dataset
fprintf('First 10 examples from the dataset: \n');
fprintf('x = [%.0f %.0f], y = %0.f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
fprintf('\n');
pause;

% Scale Feature and set them to zero mean
fprintf('Normalizing Features ... \n');

function [X_norm, mu, sigma] = featureNormalize(X)
  % FeatureNormalize(X) returns a normalize version of X where 
  % the mean value of each feature is 0 and the standard deviation is 1.
  % This is often a good prepossing step to do when working with learning algorithm. 
  mu = mean(X);
  sigma = std(X);
  X_norm = (X.-mu)./sigma
end

[X mu sigma] = featureNormalize(X);

% Add intercept term to X (i.,e. adding 1 to entire column of x0)
X = [ones(m, 1), X];

%% ========================================================
%% ========== Part 2: Cost J & Gradient Descent ===========
%% ========================================================
% Before performing gradient descent we have to compute cost J 

% Calculating Cost J
function J = computeCostMulti(X, y, theta)
  % J = computeCostMulti(X, y, theta) computes the cost of using theta
  % as the parameter for linear regression to fit the data points in X and y
  m = length(y); % number of training examples
  J = (1/(2*m))*sum(((X*theta) - y).^2) % formula for cost function
end

% Perform Gradient Descent to update theta
function[theta, J_history] = gradientDescentMulti(X, y,theta, alpha, num_iters)
  m = length(y);
  
  for iter = 1:num_iters
    theta = theta - (alpha/m)*(X')*(X*theta - y);
    J_history(iter) = computeCostMulti(X, y, theta); 
  endfor
end

fprintf('Running Gradient Descent ... \n');

% Choose appropriate value for alpha and set no. of iterations
alpha = 0.01; % 0.1 is even better
num_iters = 400;

% Initiate theta and run gradient descent
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Display the gradient discent's result
fprintf('Theta computed from Gradient Descent: \n');
fprintf('%f \n', theta);
fprintf('\n');

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iteratiuons');
ylabel('Cost J');

fprintf('Program paused. Press enter to continue.\n');
fprintf('\n');
pause;

% ====== Price Estimation ===========
% Estimate the price of a 1650 sq-ft, 3 beadroom house


% The first column of X is all ones. Thus, it does not need to be normalized.
house = [1 1650 3];
house(1,2) = (house(1,2) - mu(1,1)) / (sigma(1,1));
house(1,3) = (house(1,3) - mu(1,2)) / (sigma(1,2));
price = house * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 bedroom house' ...
        '(using Gradiernt Descent):\n $%f\n'], price );
fprintf('\n');

fprintf('Program paused. Press enter to continue.\n');
pause;
fprintf('\n');

% =================================================
% ============== Part 3: Normal Equations =========
% =================================================
fprintf('solving with Normal Equations ... \n');

function [theta] = normalEqn(X,y)
  theta = pinv(X' * X) * (X') * y; % formula for norma equation
endfunction

% Load Data
data = csvread('dataset.txt');
X = data(:, 1:2);
y = data(:,3);
m = length(y);

% Add intercept term to X (i.,e. adding 1 to entire column of x0)
X = [ones(m, 1), X];

% calculating the parameters for the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the Normal Equations: \n');
fprintf('%f \n', theta);
fprintf('\n');

price = [1 1650 3] * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 bedroom house' ...
        '(using Normal Equations):\n $%f\n'], price );
