function out = EXTRA_symmetrized(Nlist,K,alpha,lam,time_var_mat,eq_start,init,perf,fctParam,Ninf)
% Compute the worst-case performance of K steps of EXTRA for L-smooth and
% mu-strongly convex local functions, using a compact symmetrized PEP
% formulation from [1]. The size of the resulting SDP PEP depends on
% on the total number of iterations K and the number of equivalence classes of agents 
% (given by the lenght of Nlist), but not on the total number of agents in the problem.
% REQUIREMENTS: YALMIP toolbox with Mosek solver.
% INPUTS:
%   Nlist : list of length m=number of equivalence classes of agents.
%   The elements in the list give the number of agents in each class and sum to N.
%   K : number of iterations
%   alpha : step-size (constant or vector of K elements)
%   lam: matrix description (suported options)
%           The full matrix (N x N)
%           eigenvalue bound ( on the eigenvalues of the consensus matrix used in DGD
%   time_var_mat : boolean; 1 if the consensus matrix can vary across the iteration and 0 otherwise.
%   eq_start : boolean to indicate if the agents start with the same initial iterate
%   init : string to choose the initial condition to consider
%   perf : string to choose the performance criterion to consider
%   fctParam : struct with values for 'L' and 'mu'for each equivalence class of agents
%   Ninf :  if 1, compute the solution for N->infty.
%           In that case Nlist provides size proportions of each classes. 
% OUTPUT: structure with details about the worst-case solution
% of the PEP, including
%   WCperformance : worst-case value.
%   solverDetails : details given by the Mosek solver

% Source: 
%   [1] S. Colla and J. M. Hendrickx, "Exploiting Agent Symmetries for Performance Analysis of Distributed
%       Optimization Methods", 2024.

% TO DO alert on conditioning

verbose = 1;                % Print the problem set up and the results
verbose_solv = 0;           % Print details of the solver
trace_Heuristic = 0;        % heuristic to minimize the dimension of the worst-case
estimateW = 1;              % Estimate the worst averaging matrix

% Constants for initial conditions
D = 1;                     % Constant for the initial condition: ||x0 - xs||^2 <= D^2
E = 1;                     % Constant for initial condition on s_0

% Network of agents
n = sum(Nlist);
m = length(Nlist);
nu = Nlist;

% Number of agents tends to infty ?
if nargin < 10
    Ninf = 0;
end
if ~Ninf % N finite
    prop = nu/n; % proportion of agents in each class
    over_n = 1/n;
else % N -> infty
    prop = nu/n; % proportion of agents in each class
    over_n = 0;
end

switch perf
    case 'it_err_last_worst'
        assert(m == 2,"You should provide 2 subsets of agents: [1 (the worst), N-1 (the rest)]")
    case {'it_err_last_percentile_worst', 'fct_err_last_percentile_worst'}
        assert(m == 3,"You should provide 3 subsets of agents, i.e. [(1-p)N, 1, p*N-1], where p is the percentile we evaluate")
    case 'fct_err_last_worst'
        assert(m == 2,"You should provide 2 subsets of agents: [1 (the worst), N-1 (the rest)]")
end

if length(fctParam) < m % parameters of the class of functions for each subset
    fctParam = repmat(fctParam,m,1);
end

% bounds on eigenvalues
if length(lam) == 1
    lamW = [-lam,lam];
else
    lamW = lam;
end

% Defininfg the coefficient to access SDP variables easily
% Pi = [x0 g0...gK Wx0...WxK-1 (geval xeval) gs]_i 
% Fi = [f0..fK (feval)]_i  %feval2
dimG = 2*(K+1)+3;
dimF = K+2;
nbPts = K+3;
if verbose
    fprintf("Problem Size = %d \n", dimG);
end

% Optimum
xs   = zeros(dimG,1); % optimal point
%xs(end-1) = 1; % We suppose x* = 0
gs   = zeros(dimG,1); % local gradient at optimum
gs(end) = 1;
fs   = zeros(dimF,1); % local function value at optimum
%fs(end) = 1

% Iterates init
x  = zeros(dimG, K+1);  x(1,1)      = 1;
g  = zeros(dimG, K+1);  g(2:K+2,:)  = eye(K+1);
f  = zeros(dimF,K+1);   f(1:K+1,:)  = eye(K+1);

% Point common to all agents, used for Performance Evaluation
gc  = zeros(dimG,1); gc(2*(K+1)+1)= 1;
xc  = zeros(dimG,1); xc(2*(K+1)+2)= 1;
fc  = zeros(dimF,1); fc(K+2)      = 1;

% Consensus iterates
Wx = zeros(dimG, K); Wx(K+3:2*K+2,:)    = eye(K);

% EXTRA iterates
if K>0
    x(:,2) = Wx(:,1) - alpha*g(:,1);
end
for k = 1:K-1
    x(:,k+2) = x(:,k+1) + Wx(:,k+1) -1/2*(x(:,k) + Wx(:,k) ) - alpha*(g(:,k+1) - g(:,k));
end

% Interpolation concatenation
Xinterp = [xs, x, xc];
Ginterp = [gs, g, gc];
Finterp = [fs, f, fc];

Xcons = [x(:,1:K)];
Wxcons = [Wx];

%% PEP problem
% We consider m equivalence classes of Nu agents
Ga = cell(m,1); % diagonal blocks of symmetrized Gram
Gb = cell(m,1); % off-diagonal blocks of symmetrized Gram (for same agents from the same subset)
Ge = cell(m*(m-1)/2,1); % off-diagonal blocks of symmetrized Gram (for different subsets agents)
fa = cell(m,1); % Vector of function values for each block of the symmetrized F vector.

% Defining the SDP variables
for u=1:m
    Ga{u} = sdpvar(dimG);
    if (nu(u)>1 && ~Ninf) || (prop(u) > 0 && Ninf) % more than one agent in class u
        Gb{u} = sdpvar(dimG);
    else % only one agent in class u
        Gb{u} = 0;
    end
    for v=u+1:m
        Ge{u+v-2} = sdpvar(dimG,dimG,'full');
    end
    fa{u} = sdpvar(1,dimF);
end

% SDP consrtraint G>= 0
for u=1:m
    if u==1
        cons = (Ga{u}-Gb{u})>=0;
    else
        cons = cons + ((Ga{u}-Gb{u})>=0);
    end
end

H = blkvar;
for u = 1:m
    H(u,u) = prop(u)*over_n*Ga{u} + prop(u)*(prop(u)-over_n)*Gb{u};
    for v = u+1:m
        H(u,v) = prop(u)*prop(v)*Ge{u+v-2};
        %H{v,u} = nu(u)*nu(v)*Ge{u+v-2}';
        %H{v,u} filled out automatically by yalmip thanks to symmetry of blkvar
    end
end
cons = cons + (H >= 0);

% Definition of variables GA, GC and GD for easy constraint definition
 fA = 0; GA = 0; GC = 0; %GD = 0;
for u=1:m
    fA = fA + prop(u)*fa{u};
    GA = GA + prop(u)*Ga{u};
    GC = GC + prop(u)*(Ga{u}*over_n + (prop(u)-over_n)*Gb{u});
    for v=u+1:m
        GC = GC + prop(u)*prop(v)*(Ge{u+v-2}+Ge{u+v-2}');
    end
end
GD = GA - GC;

%cons = cons + (GD >= 0);
cons = cons + (GC >= 0);

% Optimality Constraint
cons = cons + (gs.'*(GC)*gs == 0); % WARNING: ill-conditionned?

% Equality of all the optimal point for all the agents
% (not needed if index of xs is 0)
if ~all(xs==0)
    cons = cons + (xs.'*(GD)*xs == 0);
end

% Def of common point xc, used in the performance criterion
switch perf %WARNING: ill-conditionned?
    case 'fct_err_Kavg_Navg' % xeval is avg_k avg_i x_i^k
        cons = cons + (xc.'*(GA)*xc + mean((x(:,:) - 2*xc).'*GC*x(:,:),'all') == 0);
    case 'fct_err_last_Navg' % xeval is avg_i x_i^K
        cons = cons + (xc.'*(GA)*xc + (x(:,K+1) - 2*xc).'*(GC)*x(:,K+1) == 0);
    case 'fct_err_last_worst' % xeval is x_1^K
        cons = cons + (xc.'*Ga{2}*xc + x(:,K+1).'*Ga{1}*x(:,K+1) -  2*xc.'*(Ge{1}')*x(:,K+1) == 0);
end

% INTERPOLATION OF local functions fi
for i = 1:nbPts
    for j = 1:nbPts
        xi = Xinterp(:,i); gi = Ginterp(:,i); fi = Finterp(:,i);
        xj = Xinterp(:,j); gj = Ginterp(:,j); fj = Finterp(:,j);
        if i ~= j
            for u=1:m
                L = fctParam(u).L;  mu = fctParam(u).mu; 
                cons = cons + ( (fa{u}*(fj-fi) + gj.'*(Ga{u})*(xi-xj) + ...
                    1/(2*(1-mu/L)) *(1/L*((gi-gj).'*(Ga{u})*(gi-gj)+...
                    mu *( (xj - xi).'*(Ga{u})*(xj - xi))  - ...
                    2*mu/L*( (xj - xi).'*(Ga{u})*(gj - gi) ) )  )) <= 0 );
            end
        end
    end
end

% INTERPOLATION of communication network - Constraints for consensus Y = WX
% Average preserving
cons = cons + ( (Xcons-Wxcons).'*(GC)*(Xcons-Wxcons) == 0); % WARNING: ill-conditionned?
% Spectral conditions
if ~time_var_mat
    cons = cons + ( (Wxcons-lamW(1)*Xcons).'*(GD)*(Wxcons-lamW(2)*Xcons) <= 0); % Wx^2 <= lam^2 x^2 (all consensus steps at once)
    if K>1
        cons = cons + (Wxcons.'*(GD)*Xcons - Xcons.'*(GD)*Wxcons == 0); %XY = YX (see beginning of carnet 2023), we can alos use GA because X = Xb + Xp
    end
else
    for k = 1:length(Wxcons(1,:))
        cons = cons + ( (Wxcons(:,k)-lamW(1)*Xcons(:,k)).'*(GD)*(Wxcons(:,k)-lamW(2)*Xcons(:,k)) <= 0); % Wx^2 <= lam^2 x^2 (each consensus step independently)
    end
end

% INITIAL CONDITION
if eq_start
    cons = cons + (x(:,1).'*(GD)*x(:,1) == 0); % same starting point % WARNING: ill-conditionned?
end

switch init.type
    case 'diging_like'
        for u=1:m
            % ||x_i^k - xs||^2 <= D^2 for all i \in class u
            cons = cons + ((x(:,1)-xs).'*(Ga{u})*(x(:,1)-xs) <= D^2);   % bound on distance of starting point to optimum for each class (QST: why not for all class?)
        end
        cons = cons + (g(:,1).'*(GD)*g(:,1) <= E^2);                    % bound on the distance of the initial gradient to their average: 
                                                                        % avg_i ||g_i(x_i^0) - avg_j(g_j(x_j^0))||^2 <= E^2
    case 'diging_like_combined' % summing all the constraints in one
        cons = cons + (x(:,1).'*(GA)*x(:,1) + init.gamma*(g(:,1).'*(GD)*g(:,1)) <= D^2);
    case 'uniform_bounded_iterr_local_grad0'
        for u=1:m
            % ||x_i^k - xs||^2 <= D^2 for all i \in class u
            cons = cons + ((x(:,1)-xs).'*(Ga{u})*(x(:,1)-xs) <= D^2);   % bound on distance of starting point to optimum for each class
            cons = cons + (g(:,1).'*(Ga{u})*g(:,1) <= E^2);             % ||g_i(x_i^0)||^2 >= E^2, for all i \in class u
        end
    case 'bounded_avg_iterr_local_grad0'
        % avg_i ||x_i^k - xs||^2 <= D^2
        cons = cons + ((x(:,1)-xs).'*(GA)*(x(:,1)-xs) <= D^2);          % bound on the average distance of starting points to optimum
        cons = cons + (g(:,1).'*(GA)*g(:,1) <= E^2);                    % avg_i ||g(x_i^0)||^2 >= E^2
    case 'uniform_bounded_iterr_local_grad*'
        for u=1:m
            cons = cons + ((x(:,1)-xs).'*(Ga{u})*(x(:,1)-xs) <= D^2);   % bound on distance of starting point to optimum for each class
            cons = cons + (gs.'*(Ga{u})*gs <= E^2);                     % ||g_i(x*)||^2 >= E^2, for all i \in class u
        end
    case 'bounded_avg_iterr_local_grad*'
        % avg_i ||x_i^k - xs||^2 <= D^2
        cons = cons + ((x(:,1)-xs).'*(GA)*(x(:,1)-xs) <= D^2);          % bound on the average distance of starting points to optimum
        cons = cons + (gs.'*(GA)*gs <= E^2);                            % avg_i ||g_i(x*)||^2 >= E^2,
    otherwise %'uniform_bounded_iterr_local_grad*'
        for u=1:m
            cons = cons + ((x(:,1)-xs).'*(Ga{u})*(x(:,1)-xs) <= D^2);   % bound on distance of starting point to optimum for each class
            cons = cons + (gs.'*(Ga{u})*gs <= E^2);                     % avg_i ||g_i(x*)||^2 >= E^2,
        end 
end

% OBJECTIVE FUNCTION of the PEP
switch perf
    case 'Navg_last_it_err' % 1/N sum_i ||x_i^K - x*||2
        obj = (x(:,K+1)-xs).'*GA*(x(:,K+1)-xs);
    case 'Kavg_Navg_it_err' % avg_i avg_k ||x_i^k - x*||2
        obj = 0;
        for k = 1:K+1
            obj = obj + (x(:,k)-xs).'*GA*(x(:,k)-xs)/(K+1);
        end
    case 'it_err_last_Navg' % ||avg_i x_i^K - x*||^2
        obj = (x(:,K+1)-xs).'*GC*(x(:,K+1)-xs);
    case 'it_err_Kavg_Navg' % ||avg_i avg_k x_i^k - x*||^2
        obj = (mean(x(:,:),2)-xs).'*GC*(mean(x(:,:),2)-xs);
    case 'it_err_last_worst' % max_i ||x_i^K - x*||2
        obj = (x(:,K+1)-xs).'*Ga{1}*(x(:,K+1)-xs);
    case 'it_err_last_percentile_worst' % max_{i \in sets 2 and 3} ||x_i^K - x*||2
        obj = (x(:,K+1)-xs).'*Ga{2}*(x(:,K+1)-xs);
        obj_exclude = (x(:,K+1)-xs).'*Ga{1}*(x(:,K+1)-xs);
        cons = cons + (obj_exclude >= obj);
    case 'Navg_last_it_err_combined_with_g' % for rate in DIGing
        obj = x(:,K+1).'*(GA)*x(:,K+1) + init.gamma*(g(:,K+1).'*GD*g(:,K+1));
    case { 'fct_err_last_Navg', 'fct_err_Kavg_Navg'}
        %  F(xb(K)) - F(x*) or F(avg_k xb(k)) - F(x*)
        obj = fA*(fc-fs);
    case 'fct_err_last_worst' %max_i F(xi) - F(x*)
        obj = fa{1}*(f(:,K+1)-fs)*prop(1);
        for u = 2:m
            obj = obj + fa{u}*(fc-fs)*prop(u);
        end
    otherwise % default: Navg_last_it_err
        obj = (x(:,K+1)-xs).'*GA*(x(:,K+1)-xs);
end

% RESOLUTION of the SDP PEP
solver_opt      = sdpsettings('solver','mosek','verbose',verbose_solv);
solverDetails   = optimize(cons,-obj,solver_opt);

% OUTPUT
out.solverDetails = solverDetails;
out.WCperformance = double(obj);
out.GD = double(GD);
out.GT = double(GC);
out.GA = double(GA);
if verbose
    fprintf("Solver output %7.5e, \t Solution status %s \n",out.WCperformance, out.solverDetails.info);
end

% Trace Heuristic
if trace_Heuristic
    cons = cons + (obj >= out.WCperformance-1e-5);
    solverDetails  = optimize(cons,trace(GA),solver_opt);
    
    wc = double(obj);
    if verbose
        fprintf("Solver output after Trace Heurisitc: %7.5e, \t Solution status %s \n",wc, solverDetails.info);
    end
end



if estimateW
    % Composition of G
    G = zeros(n*dimG);
    Nc = 0;
    for u=1:m
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
        for v=u+1:m
            Guv = zeros(nu(u)*dimG,nu(v)*dimG);
            for i=0:nu(u)-1
                for j=0:nu(v)-1
                    %fprintf('v=%d, i=%d,j=%d \n',v,i,j)  
                    Guv(i*dimG+1:(i+1)*dimG,j*dimG+1:(j+1)*dimG) = double(Ge{u+v-2});
                    %Guv(j*dimG+1:(j+1)*dimG,i*dimG+1:(i+1)*dimG) = Gr{u+v-2}';
                end
            end
            G(Nc*dimG+1:(Nc+nu(u))*dimG,(Nc+nu(u))*dimG+1:(Nc+nu(u)+nu(v))*dimG) = Guv;
            G((Nc+nu(u))*dimG+1:(Nc+nu(u)+nu(v))*dimG,Nc*dimG+1:(Nc+nu(u))*dimG) = Guv'; 
        end
        Nc = Nc + nu(u);
    end

    % Factorization of G
    [V,D]=eig(double(G));%
    tol=1e-5; %Throw away eigenvalues smaller that tol
    eigenV=diag(D); eigenV(eigenV < tol)=0;
    new_D=diag(eigenV); [~,P]=qr(sqrt(new_D)*V.');
    P=P(1:sum(eigenV>0),:);
    P = double(P);
    d = length(P(:,1)) % dimension of the worst-case
    
    % Extracting X and Y
    Pi = cell(n,1);
    X_fl = zeros(n,(K)*d);
    Y_fl = zeros(n,(K)*d);
    Nc = 0;
    for u=1:m
        for i=1:nu(u)
            Pi{Nc+i} = P(:,Nc*dimG+1:(Nc+1)*dimG);
            X_fl(Nc+i,:) = reshape(Pi{Nc+i}*Xcons(:,:),[1,(K)*d]);
            Y_fl(Nc+i,:) = reshape(Pi{Nc+i}*Wxcons(:,:),[1,(K)*d]);
        end
        Nc = Nc+nu(u);
    end
    out.X = X_fl;
    out.Y = Y_fl;
    
    % Estimating worst W
    [out.Wh,out.r,out.status] = cons_matrix_estimate([-lam,lam],X_fl,Y_fl,n);
end
end