function out = EXTRA_symmetrized(Settings) %Nlist,K,alpha,lam,time_var_mat,eq_start,init,perf,fctParam,Ninf)
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

verbose = 1;            % Print the problem set up and the results
verbose_solv = 0;       % Print details of the solver
trace_heuristic = 0;    % heuristic to minimize the dimension of the worst-case
eval_out = 0;           % evaluate the worst-case local iterates and gradients and add them to the output
estim_W = 0;            % Estimate the worst averaging matrix


%%% Set up performance estimation settings %%%
if nargin == 1
    Settings = extractSettings(Settings);
else
    warning("settings should be provided in a single structure - default settings used")
    Settings = extractSettings(struct());
end
nlist=Settings.nlist; ninf=Settings.ninf; t=Settings.t; alpha=Settings.alpha;
type=Settings.type; lamW=Settings.avg_mat; tv_mat=Settings.tv_mat;
eq_start=Settings.eq_start; init=Settings.init; perf=Settings.perf;
fctClass=Settings.fctClass; fctParam=Settings.fctParam;

assert(strcmp(fctClass,'SmoothStronglyConvex'),"EXTRA_symmetrized only applies to 'SmoothStronglyConvex' (possibly with mu = 0) functions");
% other classes of function requires different interpolation conditions
assert(strcmp(type,'spectral_relaxed'),"EXTRA_symmetrized only applies to spectral description of the averaging matrix (range of eigenvalues)");

if verbose
    fprintf("Settings provided for the PEP:\n");
    fprintf('nlist = ['); fprintf('%g ', nlist); fprintf('] ');
    fprintf("ninf=%d, t=%d, alpha=%1.2f, type=%s, tv_mat=%d, eq_start=%d,\ninit_x=%s, init_grad=%s, perf=%s, fctClass=%s,\n",ninf,t,alpha(1),type,tv_mat,eq_start,init.x,init.grad,perf,fctClass);
    fprintf('avg_mat = ['); fprintf('%g ', lamW); fprintf(']\n');
    fprintf("------------------------------------------------------------------------------------------\n");
end

% Network of agents
n = sum(nlist);    % total number of agents
r = length(nlist); % number of equivalence classes of agents
nu = nlist;        % list to indicate the number of agents in each class

switch perf
    case {'it_err_last_worst','fct_err_last_worst'}
        assert(r == 2,"You should provide 2 subsets of agents: [1 (the worst), N-1 (the rest)]")
    case {'it_err_last_percentile_worst', 'fct_err_last_percentile_worst'}
        assert(r == 3,"You should provide 3 subsets of agents, i.e. [(1-p)N, 1, p*N-1], where p is the percentile we evaluate")
end

if ~ninf % n finite
    prop = nu/n; % proportion of agents in each class
    over_n = 1/n;
else % n -> infty
    prop = nu/n; % proportion of agents in each class
    over_n = 0;
end

% parameters of the class of functions for each agent equivalence class
if length(fctParam) < r
    fctParam = repmat(fctParam,r,1);
end

%% Defining the coefficient vectors to access SDP variables easily
% Gram matrix G = P^TP with P = [P1 ... Pn]
% Pi = [x0 g0...gt Wx0...Wxt-1 (geval xeval) gs]_i
% coefficient x^k is such that Pi x^k = x_i^k

% Vector of function values F = [F1 .. Fn]
% Fi = [f0..ft (feval)]_i
% coefficient f^k is such that Fi f^k = f_i^k


dimG = 2*(t+1)+3; % dimension of the Gram matrix
dimF = t+2;  % dimension of the vector of function values
nbPts = t+3; % number of points to use for the interpolation of the local functions

if verbose
    fprintf("Problem Size = %d x %d \n", dimG,dimG);
end

% Optimum coeff
xs   = zeros(dimG,1); % optimal point
%xs(end-1) = 1; % no need because we suppose x* = 0
gs   = zeros(dimG,1); % local gradient at optimum
gs(end) = 1;
fs   = zeros(dimF,1); % local function value at optimum
%fs(end) = 1    % no need because we suppose f* = 0

% Initial iterates coeff
x  = zeros(dimG, t+1);  x(1,1)      = 1;
g  = zeros(dimG, t+1);  g(2:t+2,:)  = eye(t+1);
f  = zeros(dimF,t+1);   f(1:t+1,:)  = eye(t+1);

% coeff for the point common to all agents, used in the performance criterion
gc  = zeros(dimG,1); gc(2*(t+1)+1)= 1;
xc  = zeros(dimG,1); xc(2*(t+1)+2)= 1;
fc  = zeros(dimF,1); fc(t+2)      = 1;

% Consensus iterates coeff
Wx = zeros(dimG, t); Wx(t+3:2*t+2,:)    = eye(t);

% EXTRA iterates coeff
if t>0
    x(:,2) = Wx(:,1) - alpha(1)*g(:,1);
end
for k = 1:t-1
    x(:,k+2) = x(:,k+1) + Wx(:,k+1) -1/2*(x(:,k) + Wx(:,k) ) - alpha(k)*(g(:,k+1) - g(:,k));
end

% set of points to interpolate for the local functions
Xinterp = [xs, x, xc];
Ginterp = [gs, g, gc];
Finterp = [fs, f, fc];

% set of pair of points to interpolate for the averaging matrix
Xcons = [x(:,1:t)];
Wxcons = [Wx];

%% PEP problem

% (1) VARIABLES
% We consider r equivalence classes of nu agents
Ga = cell(r,1); % diagonal blocks of symmetrized Gram
Gb = cell(r,1); % off-diagonal blocks of symmetrized Gram (for same agents from the same subset)
Ge = cell(r*(r-1)/2,1); % off-diagonal blocks of symmetrized Gram (for different subsets agents)
fa = cell(r,1); % Vector of function values for each block of the symmetrized F vector.

% Defining the SDP variables
for u=1:r
    Ga{u} = sdpvar(dimG);
    if (nu(u)>1 && ~ninf) || (prop(u) > 0 && ninf) % more than one agent in class u
        Gb{u} = sdpvar(dimG);
    else % only one agent in class u
        Gb{u} = 0;
    end
    for v=u+1:r
        Ge{u+v-2} = sdpvar(dimG,dimG,'full');
    end
    fa{u} = sdpvar(1,dimF);
end

% (2) SDP constraint G>= 0
for u=1:r
    if u==1
        cons = (Ga{u}-Gb{u})>=0;
    else
        cons = cons + ((Ga{u}-Gb{u})>=0);
    end
end

H = blkvar;
for u = 1:r
    H(u,u) = prop(u)*over_n*Ga{u} + prop(u)*(prop(u)-over_n)*Gb{u};
    for v = u+1:r
        H(u,v) = prop(u)*prop(v)*Ge{u+v-2};
        %H{v,u} filled out automatically by yalmip thanks to symmetry of blkvar
    end
end
cons = cons + (H >= 0);

% Definition of variables GA, GC and GD for easy constraints definition
fA = 0; % 1/n sum_u nu fa{u}
GA = 0; % 1/n sum_u nu Ga{u}
GC = 0; % 1/n^2 sum_u nu (Ga{u}+(nu-1)Gb{u}) + 1/n^2 sum_u sum_v nu*nv*Ge{u,v}
for u=1:r
    fA = fA + prop(u)*fa{u};
    GA = GA + prop(u)*Ga{u};
    GC = GC + prop(u)*(Ga{u}*over_n + (prop(u)-over_n)*Gb{u});
    for v=u+1:r
        GC = GC + prop(u)*prop(v)*(Ge{u+v-2}+Ge{u+v-2}');
    end
end
GD = GA - GC;

% (3) OPTIMALITY Constraint
cons = cons + (gs.'*(GC)*gs == 0); % WARNING: equality constraints deteriorate the problem conditioning

% Equality of all the optimal point for all the agents
% (not needed if index of xs is 0)
if ~all(xs==0)
    cons = cons + (xs.'*(GD)*xs == 0); % WARNING: equality constraints deteriorate the problem conditioning
end

% (4) INTERPOLATION conditions for local functions fi
for i = 1:nbPts
    for j = 1:nbPts
        xi = Xinterp(:,i); gi = Ginterp(:,i); fi = Finterp(:,i);
        xj = Xinterp(:,j); gj = Ginterp(:,j); fj = Finterp(:,j);
        if i ~= j
            for u=1:r
                L = fctParam(u).L;  mu = fctParam(u).mu;
                % condition for L-smooth and mu-strongly convex functions
                cons = cons + ( (fa{u}*(fj-fi) + gj.'*(Ga{u})*(xi-xj) + ...
                    1/(2*(1-mu/L)) *(1/L*((gi-gj).'*(Ga{u})*(gi-gj)+...
                    mu *( (xj - xi).'*(Ga{u})*(xj - xi))  - ...
                    2*mu/L*( (xj - xi).'*(Ga{u})*(gj - gi) ) )  )) <= 0 );
            end
        end
    end
end

% (5) INTERPOLATION conditions for the averaging matrix W, used in consensus steps
% Average preserving: avg_i Wxi = avg_i Xi = Xb
cons = cons + ( (Xcons-Wxcons).'*(GC)*(Xcons-Wxcons) == 0); % WARNING: equality constraints deteriorate the problem conditioning

if ~tv_mat % time-constant matrix W
    % Spectral conditions
    cons = cons + ( (Wxcons-lamW(1)*Xcons).'*(GD)*(Wxcons-lamW(2)*Xcons) <= 0); % (Wx-Xb)^2 <= lam^2 (X-Xb)^2 (all consensus steps at once)
    if t>1 % Symmetry condition
        cons = cons + (Wxcons.'*(GD)*Xcons - Xcons.'*(GD)*Wxcons == 0); %(X-Xb)^T(Wx-Xb) = (Wx-Xb)^T(X-Xb). We can also use GA (instead of GD)
    end
else % time-varying matrix W^k
    for k = 1:length(Wxcons(1,:))
        % Spectral conditions
        cons = cons + ( (Wxcons(:,k)-lamW(1)*Xcons(:,k)).'*(GD)*(Wxcons(:,k)-lamW(2)*Xcons(:,k)) <= 0); % (Wx-xb)^2 <= lam^2 (x-xb)^2 (each consensus step independently)
    end
end

% (6) INITIAL CONDITIONS
if eq_start % same starting point
    cons = cons + (x(:,1).'*(GD)*x(:,1) == 0);  % WARNING: equality constraints deteriorate the problem conditioning
end

% Initial condition for x0
switch init.x
    case {'bounded_navg_it_err','0'}      % avg_i ||xi0 - xs||^2 <= D^2
        cons = cons + ((x(:,1)-xs).'*(GA)*(x(:,1)-xs) <= init.D^2);
    case {'uniform_bounded_it_err','1'}   %||xi0 - xs||^2 <= D^2 for all i
        for u=1:r % for each class u
            cons = cons + ((x(:,1)-xs).'*(Ga{u})*(x(:,1)-xs) <= init.D^2);
        end
    case {'navg_it_err_combined_grad','2'}
        % (avg_i ||xi0 - xs||^2) + gamma* (avg_i ||s0 - avg_i(gi0)||^2) <= D^2
        cons = cons + (x(:,1).'*(GA)*x(:,1) + init.gamma*(g(:,1).'*(GD)*g(:,1)) <= init.D^2);
    otherwise % default (for EXTRA) is 'uniform_bounded_it_err'
        for u=1:r % for each class u
            cons = cons + ((x(:,1)-xs).'*(Ga{u})*(x(:,1)-xs) <= init.D^2);
        end
end

% Initial condition for g0
switch init.grad
    case {'bounded_navg_grad0','0'}       % avg_i ||gi0||^2 <= E^2
        cons = cons + (g(:,1).'*(GA)*g(:,1) <= init.E^2);
    case {'uniform_bounded_grad0','1'}    % ||gi0||^2 <= E^2 for all i
        for u=1:r % for each class u
            cons = cons + (g(:,1).'*(Ga{u})*g(:,1) <= init.E^2);
        end
    case {'bounded_grad0_cons_err','2'}    % avg_i ||gi0 - avg_i(gi0)||^2 <= E^2
        % bound on the distance of the initial gradient to their average:
        cons = cons + (g(:,1).'*(GD)*g(:,1) <= init.E^2);
    case {'bounded_navg_grad*','3'}       % avg_i ||gi(x*)||^2 <= E^2
        cons = cons + (gs.'*(GA)*gs <= init.E^2);
    case {'uniform_bounded_grad*','4'}    % ||gi(x*)||^2 <= E^2 for all i
        for u=1:r % for each class u
            cons = cons + (gs.'*(Ga{u})*gs <= init.E^2);
        end
    otherwise % default (for EXTRA) is 'uniform_bounded_grad*'
        if ~strcmp(init.x,'navg_it_err_combined_grad') && ~strcmp(init.x,'2')
            for u=1:r % for each class u
                cons = cons + (gs.'*(Ga{u})*gs <= init.E^2);
            end
        end
end

% (7) OBJECTIVE FUNCTION of the PEP
% definition of common point xc, used in the performance criterion
switch perf % WARNING: equality constraints deteriorate the problem conditioning
    case {'fct_err_last_navg','6'} % xeval is avg_i x_i^t
        cons = cons + (xc.'*(GA)*xc + (x(:,t+1) - 2*xc).'*(GC)*x(:,t+1) == 0);
    case {'fct_err_last_worst','7'} % xeval is x_1^t
        cons = cons + (xc.'*Ga{2}*xc + x(:,t+1).'*Ga{1}*x(:,t+1) -  2*xc.'*(Ge{1}')*x(:,t+1) == 0);
    case {'fct_err_tavg_navg','8'} % xeval is avg_k avg_i x_i^k
        cons = cons + (xc.'*(GA)*xc + mean((x(:,:) - 2*xc).'*GC*x(:,:),'all') == 0);
end
% definition of the performance criterion
switch perf
    case {'navg_last_it_err','0'} % 1/n sum_i ||x_i^t - x*||2
        obj = (x(:,t+1)-xs).'*GA*(x(:,t+1)-xs);
    case {'tavg_navg_it_err','1'} % avg_i avg_k ||x_i^k - x*||2
        obj = 0;
        for k = 1:t+1
            obj = obj + (x(:,k)-xs).'*GA*(x(:,k)-xs)/(t+1);
        end
    case {'navg_it_err_combined_grad','2'}
        obj = x(:,t+1).'*(GA)*x(:,t+1) + init.gamma*(g(:,t+1).'*GD*g(:,t+1));
    case {'it_err_last_navg','3'} % ||avg_i x_i^t - x*||^2
        obj = (x(:,t+1)-xs).'*GC*(x(:,t+1)-xs);
    case {'it_err_tavg_navg','4'} % ||avg_i avg_k x_i^k - x*||^2
        obj = (mean(x(:,:),2)-xs).'*GC*(mean(x(:,:),2)-xs);
    case {'it_err_last_worst','5'} % max_i ||x_i^t - x*||2
        obj = (x(:,t+1)-xs).'*Ga{1}*(x(:,t+1)-xs);
    case { 'fct_err_last_navg', 'fct_err_tavg_navg','6','8'}
        %  F(xb(t)) - F(x*) or F(avg_k xb(k)) - F(x*)
        obj = fA*(fc-fs);
    case {'fct_err_last_worst','7'} %max_i F(xi) - F(x*)
        obj = fa{1}*(f(:,t+1)-fs)*prop(1);
        for u = 2:r
            obj = obj + fa{u}*(fc-fs)*prop(u);
        end
    case {'it_err_last_percentile_worst','9'} % max_{i \in sets 2 and 3} ||x_i^t - x*||2
        obj = (x(:,t+1)-xs).'*Ga{2}*(x(:,t+1)-xs);
        obj_exclude = (x(:,t+1)-xs).'*Ga{1}*(x(:,t+1)-xs);
        cons = cons + (obj_exclude >= obj);
    otherwise % default: 'navg_last_it_err' 
        obj = (x(:,t+1)-xs).'*GA*(x(:,t+1)-xs); % 1/n sum_i ||x_i^t - x*||2
end

% (7) Solve the SDP PEP
solver_opt      = sdpsettings('solver','mosek','verbose',verbose_solv);
solverDetails   = optimize(cons,-obj,solver_opt);

if verbose
    fprintf("Solver output %7.5e, \t Solution status %s \n",double(obj), solverDetails.info);
end

% Trace Heuristic
if trace_heuristic
    cons = cons + (obj >= out.WCperformance-1e-5);
    solverDetails  = optimize(cons,trace(GA),solver_opt);
    if verbose
        fprintf("Solver output after Trace Heurisitc: %7.5e, \t Solution status %s \n",double(obj), solverDetails.info);
    end
end

% OUTPUT
out.solverDetails = solverDetails;
out.WCperformance = double(obj);
out.GD = double(GD);
out.GT = double(GC);
out.GA = double(GA);
out.Settings = Settings;

% Evaluate the worst-case iterates (for a given number of agents)
if eval_out || estim_W
    % Composition of the full Gram matrix G
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
                    %Guv(j*dimG+1:(j+1)*dimG,i*dimG+1:(i+1)*dimG) = Gr{u+v-2}';
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
    X_fl = zeros(n,(t)*d);
    Y_fl = zeros(n,(t)*d);
    Nc = 0;
    for u=1:r
        for i=1:nu(u)
            Pi{Nc+i} = P(:,Nc*dimG+1:(Nc+1)*dimG);
            X_fl(Nc+i,:) = reshape(Pi{Nc+i}*Xcons(:,:),[1,(t)*d]);
            Y_fl(Nc+i,:) = reshape(Pi{Nc+i}*Wxcons(:,:),[1,(t)*d]);
        end
        Nc = Nc+nu(u);
    end
    out.X = X_fl;
    out.Y = Y_fl;
end

% (8) (Try to) Recover the worst-case averaging matrix that links the solutions X and Y
if estim_W
    [Wh.W,Wh.r,Wh.status] = cons_matrix_estimate(lamW,X_fl,Y_fl,n);
    if verbose && strcmp(type,'spectral_relaxed')
        fprintf("The estimate of the worst-case averaging matrix is ")
        Wh.W
        %other info that could be printed : eig(Wh.W); Wh.r; Wh.status;
    end
    out.Wh = Wh;
end
end


