% Author: Jose Reinaldo da Cunha S. A. V. S. Neto
% Universidade de Brasilia
% Lasso coordinate descent implementation for NARMAX training
%
clear all
clc
%--------------------------------------------------------------------------
% Real model
N = 100;
mu = 0;
sigma = 0.04^2;
% Training data
y(1:2)=0;
u = normrnd(0,1,[1,N]);
e = normrnd(mu,sigma,[1,N]);
% Validation data
y_val(1:2)=0;
u_val = normrnd(0,1,[1,N]);
e_val = normrnd(mu,sigma,[1,N]);
y_val = y_val';
u_val = u_val';
e_val = e_val';
% Generating outputs for training and validation
for k=3:N
    y(k) = 0.5 * y(k-1) + u(k-2) + 0.1 * (u(k-2)^2) + 0.5 * e(k-1) + 0.1 * u(k-1) * e(k-2) + e(k);
    y_val(k) = 0.5 * y_val(k-1) + u_val(k-2) + 0.1 * (u_val(k-2)^2) + 0.5 * e_val(k-1) + 0.1 * u_val(k-1) * e_val(k-2) + e_val(k);
end
if(isrow(y))
    y = y';
end
if(isrow(u))
    u = u';
end
if(isrow(e))
    e = e';
end
%--------------------------------------------------------------------------
% NARMAX model
ny = 1;
nu = 2;
ne = 2;
nl = 2;
narmax = NARMAX(ny, nu, ne, nl); % Create NARMAX model
narmax.parameters = zeros(size(narmax.full_model,1),1);
%--------------------------------------------------------------------------
% LASSO coordinate descent implementation
lambda = 1;
e_lasso = y(max(max(nu, ny),ne)+1:size(y,1));
iterations = 100;
tolerance = 0.001;
diff = 100;
past_theta = narmax.parameters;

P = regressor_matrix(narmax, ny, nu, ne, y, u, e);
aux = partial_regressor_matrix(1, narmax, ny, nu, ne, y, u, e);
% for i=1:iterations      % Stop condition
while diff >= tolerance
    for j=1:size(P,2)   % For each parameter theta
        P(:,j) = partial_regressor_matrix(j, narmax, ny, nu, ne, y, u, e);   % STEP 1
        narmax.parameters(j) = ( 1/(P(:,j)'*P(:,j) ) )*wthresh((e_lasso+narmax.parameters(j)*P(:,j))'*P(:,j),'s',lambda); % STEP 2
        e_lasso = y(max(max(nu, ny),ne)+1:size(y,1))-P*narmax.parameters; % STEP 3
    end
    diff = sum(abs(past_theta-narmax.parameters));
    past_theta = narmax.parameters;
    fprintf('diff %.4f\n', diff);
end

% Print of parameters and respective terms
for i=1:size(narmax.full_model,1)
   fprintf('%.4f\t\t\t\t[', narmax.parameters(i));
   for j=1:size(narmax.full_model,2)
       fprintf('%d ', narmax.full_model(i,j));
   end
   fprintf(']\n');
end

