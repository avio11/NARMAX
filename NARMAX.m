% Autor: José Reinaldo da C.S.A.V.S. Neto
% Matrícula: 14/0169148
% Function to create NARMAX's full, process and error-correlated models
function [ narmax ] = NARMAX( ny, nu, ne, nl )

% Creating NARMAX model object
narmax.parameters = [];
narmax.process_terms = [];          % Model terms independent from the error
narmax.error_correlated_terms = []; % Model terms dependent on the error
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


    for i=1:size(narmax.full_model,1)
        flag = 0;
        for j=ny+nu+1:ny+nu+ne
            if narmax.full_model(i,j)>=1
                flag = 1;
            end
        end
        if flag==1
            narmax.error_correlated_terms = [narmax.error_correlated_terms;narmax.full_model(i,:)];
        else
            narmax.process_terms = [narmax.process_terms;narmax.full_model(i,1:ny+nu)];
        end
    end

%  M = factorial(ny+nu+ne+nl)/(factorial(ny+nu+ne)*factorial(nl))-1; % Number of NARMAX terms excluding trivial all coefficients are zeroes case

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