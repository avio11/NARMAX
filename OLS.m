% Autor: José Reinaldo da C.S.A.V.S. Neto
% Matrícula: 14/0169148

function [narmax] = OLS()

%--------------------------------------------------------------------------
% Real model
N = 400
mu = 0;
sigma = 0.04^2;
y(1:2)=0;
u = normrnd(0,1,[1,N]);
e = normrnd(mu,sigma,[1,N]);
for k=3:N
    y(k) = 0.5 * y(k-1) + u(k-2) + 0.1 * (u(k-2)^2) + 0.5 * e(k-1) + 0.1 * u(k-1) * e(k-2) + e(k);
end

%--------------------------------------------------------------------------
% NARMAX model
ny = 2;
nu = 1;
ne = 1;
nl = 2;
narmax = NARMAX(ny, nu, ne, nl, y, u, e); % Definicao do modelo e regressor para treinamento


%--------------------------------------------------------------------------
% OLS training algorithm
P = narmax.P;
[W,A] = qr(P);
size(y(max(max(nu,ny),ne):size(y,2)))
g = inv(W*W')*W'*y(max(max(nu,ny),ne):size(y,2))';
narmax.parameters = linsolve(A,g);
narmax.parameters

% %--------------------------------------------------------------------------
% % Simulate FR & OSA
% m = size(narmax.full_model,1);
% n = size(narmax.full_model,2);
% 
% Y_FR(1:max(nu,ny)) = y(1:max(nu,ny));
% for i=max(nu, ny):size(y,1)
%     V_OSA = [flip(y(i-ny+1:i)) flip(u(i-nu+1:i))];
%     V_FR  = [flip(Y_FR(i-ny+1:i)) flip(u(i-nu+1:i))];
%     
%     for j=1:m
%         aux_OSA = 0;
%         aux_FR = 0;
%         for k=1:n
%             aux_OSA = aux_OSA * (V_OSA^narmax.full_model(j,k));
%             aux_FR  = aux_FR  * (V_FR ^narmax.full_model(j,k));
%         end
%         R_OSA(i) = [R_OSA aux_OSA];
%         R_FR(i)  = [R_FR aux_FR];
%     end
%     
% end

% Build of Regressor matrix for OSA & FR simulation
k=1;
for it=max(ny,nu):size(y,1) % For each moving horizon window on y/u
    v_OSA = [flip(y(it-ny+1:it)); flip(u(it-nu+1:it))];
    v_FR  = [flip(Y_FR(it-ny+1:it)); flip(u(it-nu+1:it))];
    for i = 1:size(narmax.full_model,1)
        aux_OSA = 1;
        aux_FR  = 1;
        for j=1:size(narmax.full_model,2)
            aux_OSA = aux_OSA*((v_OSA(j))^narmax.full_model(i,j));
            aux_FR  = aux_FR*((v_FR(j))^narmax.full_model(i,j));
        end
        R_OSA(k,i) = aux_OSA;
        R_FR(k,i) = aux_FR;
    end
    Y_OSA = R_OSA(k,:)*v_OSA';
    Y_FR = R_FR(k,:)*v_FR';
    k=k+1;
end

%--------------------------------------------------------------------------
% Plot
plot(Y_OSA);
hold on
plot(Y_FR);
plot(y);
