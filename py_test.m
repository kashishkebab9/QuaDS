clear; close all; clc;

%% Add project library paths if needed
% filepath = fileparts(which('py_test.m'));
% addpath(genpath(fullfile(filepath, '..','..', 'libraries','book-phys-gmm')));
% addpath(genpath(fullfile(filepath, '..','..', 'libraries','book-thirdparty')));

%% Define simple GMM
gmm.Priors = [0.6, 0.4];  % (1 x K)
gmm.Mu = [1 3;            % (2 x K)
          2 4];
gmm.Sigma(:,:,1) = [0.5 0.1; 
                    0.1 0.5];
gmm.Sigma(:,:,2) = [0.3 0; 
                    0 0.3];

x_test = [2.5; 3];  % (2 x 1)

%% Compute responsibilities manually (for Python comparison only)
K = length(gmm.Priors);
Px_k = zeros(K, 1);
for k = 1:K
    Px_k(k) = gmm.Priors(k) * mvnpdf(x_test', gmm.Mu(:,k)', gmm.Sigma(:,:,k)) + eps;
end
Pk_x = Px_k / sum(Px_k);  % Normalize

%% Define LPV-DS dynamics
A_g(:,:,1) = [1 0; 0 1];     % Identity
A_g(:,:,2) = [-1 0; 0 -1];   % Negative identity
b_g = [0.5 1.0; -0.5 -1.0];  % (2 x 2)

%% Compute LPV-DS output using your function
x_dot = lpv_ds(x_test, gmm, A_g, b_g);  % Calls posterior_probs_gmm internally

%% Display results
disp('Gamma_k(x) from MATLAB:');
disp(Pk_x);
disp('LPV-DS output from MATLAB:');
disp(x_dot);

%% Save everything for Python testing
save('test_gmm.mat', 'gmm', 'x_test', 'Pk_x', 'A_g', 'b_g', 'x_dot', '-v7');
