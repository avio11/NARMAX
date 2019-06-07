% Author: José Reinaldo da C.S.A.V.S. Neto
% Matrícula: 14/0169148
% Orthogonal Least Squares training for NARMAX model

function [narmax, y_val, Y_OSA] = OLS()

%--------------------------------------------------------------------------
% Real model
N = 100
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

%--------------------------------------------------------------------------
% NARX process OLS training
P = regressor_matrix(narmax, ny, nu, 0, y, u, e); % Regressor for NARX part of NARMAX model
[W,A] = qr(P);
g = inv(W*W')*W'*y(max(nu,ny)+1:size(y,1));
narmax.process_parameters = linsolve(A,g);
e_narx(1:max(nu,ny)) = zeros(max(nu,ny),1);
e_narx = e_narx';
e_narx = [e_narx ;y(max(nu,ny)+1:size(y))-P*narmax.process_parameters];
%--------------------------------------------------------------------------
% NARMAX full-model OLS training
P = regressor_matrix(narmax, ny, nu, ne, y, u, e_narx); % Regressor for NARMAX training model
[W,A] = qr(P);
g = inv(W*W')*W'*y(max(max(nu,ny),ne)+1:size(y,1));
narmax.parameters = linsolve(A,g);

%--------------------------------------------------------------------------
% One-step ahead simulation 
e_osa = zeros(max(max(nu,ny),ne),1);
Y_OSA(1:max(max(nu,ny),ne)) = y_val(1:max(max(nu,ny),ne));
for i=max(max(ny,nu),ne):size(y,1)-1 % For each moving horizon window on y/u
    v_aux = [flip(y_val(i-ny+1:i)); flip(u_val(i-nu+1:i)); flip(e_osa(i-ne+1:i))];
    for j = 1:size(narmax.full_model,1)
        aux = 1;
        for k=1:size(narmax.full_model,2)
            aux = aux*((v_aux(k))^narmax.full_model(j,k));
        end
        R(j) = aux;
    end
    size(R)
    size(narmax.parameters)
    Y_OSA(i+1) = R*narmax.parameters;
    e_osa(i+1) = y_val(i+1)-Y_OSA(i+1);
end
%--------------------------------------------------------------------------
% Free-run simulation
Y_FR(1:max(nu,ny)) = y_val(1:max(nu,ny));
for i=max(ny,nu):size(y,1)-1 % For each moving horizon window on y/u
    v_aux = [flip(Y_FR(i-ny+1:i)); flip(u_val(i-nu+1:i))];
    for j = 1:size(narmax.full_model,1)
        aux = 1;
        for k=1:max(nu,ny)
            aux = aux*((v_aux(k))^narmax.full_model(j,k));
        end
        R(j) = aux;
    end
    Y_FR(i+1) = R*narmax.parameters
end

%--------------------------------------------------------------------------
% Plot
plot(Y_OSA,'o')
hold on
plot(Y_FR)
plot(y_val)
title('Simulação NARMAX');
legend('OSA output', 'FR output', 'Real output');

%--------------------------------------------------------------------------
% Print of parameters and respective terms
for i=1:size(narmax.full_model,1)
   fprintf('%.4f\t\t\t\t[', narmax.parameters(i));
   for j=1:size(narmax.full_model,2)
       fprintf('%d ', narmax.full_model(i,j));
   end
   fprintf(']\n');
end
C = size(narmax.parameters,2)








