% Autor: José Reinaldo da C.S.A.V.S. Neto
% Matrícula: 14/0169148
% Build regressor for NARMAX model based on input(u), output(y) and
% error(e)

function [ P ] = partial_regressor_matrix( column, narmax, ny, nu, ne, y, u, e )

P = [];

for it=max(max(ny,nu),ne):size(y,1)-1 % For each moving horizon window on y/u
    v_aux = [flip(y(it-ny+1:it)); flip(u(it-nu+1:it)); flip(e(it-ne+1:it))];
    aux = 1;
    for j=1:size(narmax.full_model,2)
        aux = aux*((v_aux(j))^narmax.full_model(column,j));
    end
    P(it-max(max(ny,nu),ne)+1) = aux;
end
if isrow(P)
    P = P';
end


end

