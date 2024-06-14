function [G, P, X, Y] = fullSymSol(dimG,Ga,Gb,Ge,r,nu,t,Xcons,Wxcons)
    n = sum(nu);
    G = zeros(n*dimG);
    Nc = 0;
    for u=1:r
        Gu = zeros(nu(u)*dimG);
        for i=0:nu(u)-1
            Gu(i*dimG+1:(i+1)*dimG,i*dimG+1:(i+1)*dimG) = double(Ga{u});
            for j=i+1:nu(u)-1
                %fprintf('i=%d,j=%d \n',i,j)
                Gu(i*dimG+1:(i+1)*dimG,j*dimG+1:(j+1)*dimG) = double(Gb{u});
                Gu(j*dimG+1:(j+1)*dimG,i*dimG+1:(i+1)*dimG) = double(Gb{u});
            end
        end
        G(Nc*dimG+1:(Nc+nu(u))*dimG,Nc*dimG+1:(Nc+nu(u))*dimG) = Gu;
        for v=u+1:r
            Guv = zeros(nu(u)*dimG,nu(v)*dimG);
            for i=0:nu(u)-1
                for j=0:nu(v)-1
                    %fprintf('v=%d, i=%d,j=%d \n',v,i,j)
                    Guv(i*dimG+1:(i+1)*dimG,j*dimG+1:(j+1)*dimG) = double(Ge{u+v-2});
                end
            end
            G(Nc*dimG+1:(Nc+nu(u))*dimG,(Nc+nu(u))*dimG+1:(Nc+nu(u)+nu(v))*dimG) = Guv;
            G((Nc+nu(u))*dimG+1:(Nc+nu(u)+nu(v))*dimG,Nc*dimG+1:(Nc+nu(u))*dimG) = Guv';
        end
        Nc = Nc + nu(u);
    end
    
    % Factorization of the Gram matrix G
    [V,D]=eig(double(G));%
    tol=1e-5; %Throw away eigenvalues smaller that tol
    eigenV=diag(D); eigenV(eigenV < tol)=0;
    new_D=diag(eigenV); [~,P]=qr(sqrt(new_D)*V.');
    P=P(1:sum(eigenV>0),:);
    P = double(P);
    d = length(P(:,1)); % dimension of the worst-case
    
    % Extracting X and Y
    Pi = cell(n,1);
    X = zeros(n,(t)*d);
    Y = zeros(n,(t)*d);
    Nc = 0;
    for u=1:r
        for i=1:nu(u)
            Pi{Nc+i} = P(:,(Nc+i-1)*dimG+1:(Nc+i)*dimG);
            X(Nc+i,:) = reshape(Pi{Nc+i}*Xcons(:,:),[1,(t)*d]);
            Y(Nc+i,:) = reshape(Pi{Nc+i}*Wxcons(:,:),[1,(t)*d]);
        end
        Nc = Nc+nu(u);
    end
end