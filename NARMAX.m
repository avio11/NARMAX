function [ narmax ] = NARMAX( ny, nu, ne, nl, y, u, e )

if(isrow(y)) % Transform output vector into column vector if necessary
    y = y';
end
if(isrow(u)) % Transform input vector into column vector if necessary
    u = u';
end
if(isrow(e))
    e = e';
end
narmax.parameters = [];
narmax.P = [];
narmax.full_model = [];
narmax.ny = ny;
narmax.nu = nu;
narmax.ne = ne;
narmax.nl = nl;

aux = zeros(ny+nu+ne,1)';

aux(1) = 1;

% Generate all parameter combinations (input, output, error)
aux_p_index = variable_for_loop(1, zeros(1,ny+nu+ne), ny+nu+ne, nl);
for i=1:size(aux_p_index,1)
    narmax.full_model = [narmax.full_model;unique(perms(aux_p_index(i,:)), 'rows')];
end

%  M = factorial(ny+nu+ne+nl)/(factorial(ny+nu+ne)*factorial(nl))-1; % Number of NARMAX terms excluding trivial all coefficients are zeroes case

% Build of Regressor matrix for NARX model
% k=1;
% for it=max(ny,nu):size(y,1) % For each moving horizon window on y/u
%     v_aux = [flip(y(it-ny+1:it)); flip(u(it-nu+1:it))];
%     for i = 1:size(narmax.full_model,1)
%         aux = 1;
%         for j=1:size(narmax.full_model,2)
%             aux = aux*((v_aux(j))^narmax.full_model(i,j));
%         end
%         narmax.P(k,i) = aux;
%     end
%     k=k+1;
% end
k=1;
for it=max(max(ny,nu),ne):size(y,1) % For each moving horizon window on y/u
    v_aux = [flip(y(it-ny+1:it)); flip(u(it-nu+1:it)); flip(e(it-ne+1:it))];
    for i = 1:size(narmax.full_model,1)
        aux = 1;
        for j=1:size(narmax.full_model,2)
            aux = aux*((v_aux(j))^narmax.full_model(i,j));
        end
        narmax.P(k,i) = aux;
    end
    k=k+1;
end

end

% Nested FOR loops function of variable size
function [ aux ] = variable_for_loop(iteracao, estrutura, tam, nl)

aux = [];

for i=1:size(estrutura,1)
    if iteracao ~= 1
        for j=0:estrutura(i,iteracao-1)
            estrutura(i,iteracao) = j;
            if(sum(estrutura(i,:))<=nl)
                aux = [aux;estrutura(i,:)];
            end
        end
    else
        for j=1:nl
            estrutura(i,iteracao) = j;
            aux = [aux;estrutura(i,:)];
        end
    end
end
if iteracao ~= tam
    aux = variable_for_loop(iteracao+1, aux, tam, nl);
end

end