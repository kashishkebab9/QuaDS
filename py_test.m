% Recreate everything manually and save
clear; clc;

gmm.Priors = [0.6, 0.4];
gmm.Mu = [1 3; 2 4];
gmm.Sigma(:,:,1) = [0.5 0.1; 0.1 0.5];
gmm.Sigma(:,:,2) = [0.3 0; 0 0.3];

x_test = [2.5; 3];

K = length(gmm.Priors);
Px_k = zeros(K, 1);
for k = 1:K
    Px_k(k) = gmm.Priors(k) * mvnpdf(x_test', gmm.Mu(:,k)', gmm.Sigma(:,:,k)) + eps;
end
Pk_x = Px_k / sum(Px_k);

A_g(:,:,1) = [1 0; 0 1];
A_g(:,:,2) = [-1 0; 0 -1];
b_g = [0.5 1.0; -0.5 -1.0];

x_dot = zeros(2, 1);
for k = 1:K
    x_dot = x_dot + Pk_x(k) * (A_g(:,:,k) * x_test + b_g(:,k));
end

% Save explicitly using version 7
save('test_gmm.mat', 'gmm', 'x_test', 'Pk_x', 'A_g', 'b_g', 'x_dot', '-v7');

savepath = fullfile(pwd, 'test_gmm.mat');
save(savepath, 'gmm', 'x_test', 'Pk_x', 'A_g', 'b_g', 'x_dot', '-v7');
disp("Saved to:");
disp(savepath);