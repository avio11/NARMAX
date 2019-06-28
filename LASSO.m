% Author: Jose Reinaldo da Cunha S. A. V. S. Neto
% Universidade de Brasilia
% Lasso coordinate descent implementation for NARMAX training
%
clear all
clc
%--------------------------------------------------------------------------
% Real model
N = 400;
mu = 0;
sigma = 0.4^2;
% Training data
u = normrnd(0,1,[1,N]);
e = normrnd(mu,sigma,[1,N]);
y(1:2) = 0;
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
lambda = 10;
for it = 1:10
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
%             e_lasso = e_lasso - P(:,j)*(narmax.parameters(j) - past_theta(j));
        end
        diff = sum(abs(past_theta-narmax.parameters));
        past_theta = narmax.parameters;
        fprintf('diff %.4f\n', diff);
    end
end  
% Print of parameters and respective terms
fprintf('\t\t\t\t\t[');
for i=1:ny
    fprintf('\ty[k-%d]', i);
end
for i=1:nu
    fprintf('\tu[k-%d]', i);
end
for i=1:ne
    fprintf('\te[k-%d]', i);
end
fprintf(']\n');
for i=1:size(narmax.full_model,1)
   fprintf('%.4f\t\t\t\t[', narmax.parameters(i));
   for j=1:size(narmax.full_model,2)
       fprintf('\t\t%d ', narmax.full_model(i,j));
   end
   fprintf(']\n');
end


% Simulation of NARMAX (but only NARX terms)

Y_FR(1:100) = 0;
Y_OSA(1:100) = 0;

for i=max(nu, ny):size(y,1)-1
    Y_FR(i+1) = narmax.parameters(3)*u(i-1) + narmax.parameters(5)*Y_FR(i) + narmax.parameters(14)*Y_FR(i)*u(i-1) + narmax.parameters(18)*u(i-1)^2 + narmax.parameters(20)*Y_FR(i)^2;
    Y_OSA(i+1) = narmax.parameters(3)*u(i-1) + narmax.parameters(5)*y(i) + narmax.parameters(14)*y(i)*u(i-1) + narmax.parameters(18)*u(i-1)^2 + narmax.parameters(20)*y(i)^2;
end

% Plot Free-run and one-step ahead
figure(1)
plot(y, 'r');
hold on
plot(Y_FR, 'g');
plot(Y_OSA, 'k');
legend('Expected output', 'Free run', 'One-step ahead');
title('LASSO training');
hold off


% Mean square error (MSE) of FR and OSA simulations
mse_osa = ((y-Y_OSA')'*(y-Y_OSA'))/length(y);
mse_fr  = ((y-Y_FR')'*(y-Y_FR'))/length(y);
figure(2)
c = categorical({'OSA MSE','FR MSE'});
bar(c, [mse_osa mse_fr]);
title('Mean Square Error of Simulation')
