% Autor: José Reinaldo da C.S.A.V.S. Neto
% Matrícula: 14/0169148
% Build regressor for NARMAX model based on input(u), output(y) and
% error(e)

function [P] = regressor_matrix(narmax, ny, nu, ne, y, u, e)

P = [];
k=1;
if ne>0
    for it=max(max(ny,nu),ne):size(y,1)-1 % For each moving horizon window on y/u
        v_aux = [flip(y(it-ny+1:it)); flip(u(it-nu+1:it)); flip(e(it-ne+1:it))];
        for i = 1:size(narmax.full_model,1)
            aux = 1;
            for j=1:size(narmax.full_model,2)
                aux = aux*((v_aux(j))^narmax.full_model(i,j));
            end
            P(k,i) = aux;
        end
        k=k+1;
    end
else
    for it=max(ny,nu):size(y,1)-1 % For each moving horizon window on y/u
        v_aux = [flip(y(it-ny+1:it)); flip(u(it-nu+1:it))];
        for i = 1:size(narmax.process_terms,1)
            aux = 1;
            for j=1:size(narmax.process_terms,2)
                aux = aux*((v_aux(j))^narmax.process_terms(i,j));
            end
            P(k,i) = aux;
        end
        k=k+1;
    end
end
end

