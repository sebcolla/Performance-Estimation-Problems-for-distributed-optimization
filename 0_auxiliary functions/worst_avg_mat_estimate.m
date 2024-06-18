function [Wh,r,status] = worst_avg_mat_estimate(lam,X,Y,n)
%cons_matrix_estimate computes an estimation of the averaging matrix W 
% associated with the set of pair of points (X, Y) such that: Y ? WX
% INPUT :
% lam : 2 elements array with the bounds on the second largest eigenvalue
% of the matrix (in absolute value): lam(1) <= |lam_2(W)| <= lam(2).
% X : array of iterates of size n x t
% Y : array of consensus iterates of size n x t
% n : size of the consensus network.
% OUTPUT :
%   Wh : estimate of the consensus matrix
%   r : residual procuded by Wh: r = ||Wh*X - Y||_1/n;
%   status: indicates which estimate has been returned (pseudo-inverse estimate or SDP estimate).
            Wh = Y*pinv(X); % pseudo-inverse estimate
            status = "pseudo-inverse estimate";
            % check if Wh is feasible (with tolerance tol)
            tol = 1e-3;
            ev = eig(Wh); ev = ev(~(abs(ev-1)<= tol));
            check_ev = all(lam(2) - ev >= -tol) && all(lam(1) - ev <= tol);
            check_sym = all(abs(Wh - Wh') <= tol,'all');
            check_dstoch = all(abs(ones(1,n)*Wh - ones(1,n)) <= tol);
            if ~check_ev || ~check_sym || ~check_dstoch % Wh not feasible
                 % Compute another estimate based on the following SDP
                 I = eye(n,n); V = sdpvar(n); % symmetric by default
                 objective = norm(Y - V*X,'fro');
                 cons = [];
                 cons = cons + (ones(1,n)*V == ones(1,n)); % stochasticity 
                 cons = cons + (lam(2)*I - (V-1/n*ones(n,n)) >= 0); % bound eigenvalues
                 cons = cons + (lam(1)*I - (V-1/n*ones(n,n)) <= 0); % bound eigenvalues
                 solver_opt = sdpsettings('solver','mosek','verbose',0);
                 optimize(cons,objective,solver_opt);
                 Wh = double(V);
                 status = "SDP estimate";
            end
            %[s1,s2] = size(Y);
            r = norm(Wh*X - Y,1)/(n);  % maximal column average in the error
            if r <= 0.01
                status = status + " - success";
            else
                status = "No valid worst-case averaging matrix";
            end
end
