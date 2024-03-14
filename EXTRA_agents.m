function out = EXTRA_agents(N,K,alpha,lam,time_var_mat,eq_start,init,perf,fctParam)
% Compute the worst-case performance of K steps of EXTRA for L-smooth and
% mu-strongly convex local functions, using an agent-dependent PEP formulation [1].
% The size of the resulting SDP PEP depends on the total number of iterations K
% and on the number of agents N in the problem.
% REQUIREMENTS: PESTO and YALMIP toolboxes with Mosek solver.
% INPUT:
%   N : number of agents
%   K : number of iterations
%   alpha : step-size (constant or vector of K elements)
%   lam: matrix description (suported options)
%           The full matrix (N x N)
%           eigenvalue bound ( on the eigenvalues of the consensus matrix used in DGD
%   time_var_mat : boolean; 1 if the consensus matrix can vary across the
%   iteration and 0 otherwise.
%   eq_start : boolean to indicate if the agents start with the same initial iterate
%   init : string to choose the initial condition to consider
%   perf : string to choose the performance criterion to consider
%   fctParam : struct with values for 'L' and 'mu'for each equivalence class of agents
%               default values : L=1, mu=0.1;
% OUTPUT: structure with details about the worst-case solution of the PEP, including
%   WCperformance : worst-case value.
%   solverDetails : details given by the Mosek solver
%   
% Source: 
%   [1] S. Colla and J. M. Hendrickx, "Exploiting Agent Symmetries for Performance Analysis of Distributed
%       Optimization Methods", 2024.

verbose = 0;         % Print the problem set up and the results
trace_Heuristic = 0; % heuristic to minimize the dimension of the worst-case
estimateW = 1;       % Estimate the worst averaging matrix

%%% Set up general problem parameters %%%
if all(size(lam) ~= 1)
% (a) Exact formulation (fixed network W)
    type = 'exact';
    mat = lam;
elseif length(lam)==1
% (b) Spectral formulation
    type = 'spectral_relaxed';  % type of representation for the communication matrix
    mat = [-lam,lam];           % Range of eigenvalues for the symmetric (generalized) doubly stochastic communication matrix W
else % lam contains 2 differents bounds
    type = 'spectral_relaxed';  % type of representation for the communication matrix
    mat = lam;                  % Range of eigenvalues for the symmetric (generalized) doubly stochastic communication matrix W
end

% Constants for initial conditions
D = 1;                      % Constant for the initial condition: ||x0 - xs||^2 <= D^2
E = 1;                      % Constant for initial condition on s_0

% (0) Initialize an empty PEP
P = pep();   

% (1) Set up the local and global objective functions
fctClass = 'SmoothStronglyConvex'; % Class of functions to consider for the worst-case
if nargin < 9
    fctParam.L  = 1;
    fctParam.mu = 0.1;
end
returnOpt = 0;                      % do you need the optimal point of each local function ?
[Fi,Fav,~,~] = P.DeclareMultiFunctions(fctClass,fctParam,N,returnOpt);
[xs,Fs] = Fav.OptimalPoint(); 
%P.AddConstraint(xs^2 == 0); %P.AddConstraint(Fs == 0);

% Iterates cells
X = cell(N, K+1); Wx = cell(N, K);   % local iterates
F_saved = cell(N,K);
G_saved = cell(N,K);

% (2) Set up the starting points and initial conditions
X(:,1) = P.MultiStartingPoints(N,eq_start);
[G_saved(:,1),F_saved(:,1)] = LocalOracles(Fi,X(:,1));

switch init.type
    case 'diging_like'
        P.AddConstraint(1/N*sumcell(foreach(@(xi) (xi-xs)^2,X(:,1))) <= D^2);   % avg_i ||xi0 - xs||^2 <= D^2
        P.AddConstraint(1/N*sumcell(foreach(@(gi) (gi - 1/N*sumcell(G_saved(:,1)))^2,G_saved(:,1))) <= E^2); % avg_i ||gi0 - avg_i(gi0)||^2 <= E^2
    case 'diging_like_combined'
        metric = 1/N*sumcell(foreach(@(x0, g0)(x0-xs)^2 + init.gamma*(g0 - 1/N*sumcell(G_saved(:,1)))^2,X(:,1), G_saved(:,1)));
        P.AddConstraint(metric <= D^2);
    case 'uniform_bounded_iterr_local_grad0'
        P.AddMultiConstraints(@(xi) (xi-xs)^2 <= D^2, X(:,1));                  % ||xi0 - xs||^2 <= D^2 for all i
        P.AddMultiConstraints(@(gi) gi^2 <= E^2, G_saved(:,1));                 % ||gi0||^2 <= E^2 for all i
    case 'bounded_avg_iterr_local_grad0'
        P.AddConstraint(1/N*sumcell(foreach(@(xi) (xi-xs)^2,X(:,1))) <= D^2);   % avg_i ||xi0 - xs||^2 <= D^2
        P.AddConstraint(1/N*sumcell(foreach(@(gi) gi^2, G_saved(:,1))) <= E^2); % avg_i ||gi0||^2 <= E^2
    case 'uniform_bounded_iterr_local_grad*'
        P.AddMultiConstraints(@(xi) (xi-xs)^2 <= D^2, X(:,1));                  % ||xi0 - xs||^2 <= D^2 for all i
        [Gis,~] = LocalOracles(Fi,repmat({xs},N,1));
        P.AddMultiConstraints(@(gi) gi^2 <= E^2, Gis);                          % ||gi(x*)||^2 <= E^2 for all i
    case 'bounded_avg_iterr_local_grad*'
        P.AddConstraint(1/N*sumcell(foreach(@(xi) (xi-xs)^2,X(:,1))) <= D^2);   % avg_i ||xi0 - xs||^2 <= D^2
        [Gis,~] = LocalOracles(Fi,repmat({xs},N,1));
        P.AddConstraint(1/N*sumcell(foreach(@(gi) gi^2, Gis)) <= E^2);          % avg_i ||gi(x*)||^2 <= E^2
    otherwise %'uniform_bounded_iterr_local_grad*'
        P.AddConstraint(1/N*sumcell(foreach(@(xi) (xi-xs)^2,X(:,1))) <= D^2);   % avg_i ||xi0 - xs||^2 <= D^2
        [Gis,~] = LocalOracles(Fi,repmat({xs},N,1));
        P.AddMultiConstraints(@(gi) gi^2 <= E^2, Gis);                          % ||gi(x*)||^2 <= E^2 for all i
end
                
% (3) Set up the communication matrix
W = P.DeclareConsensusMatrix(type,mat,time_var_mat);

% (4) Algorithm (DGD)
% Step-size 
if length(alpha) == 1 % use the same step-size for each iteration
    alpha = alpha*ones(1,K);
end

% Iterations
if K > 0
    Wx(:,1) = W.consensus(X(:,1));
    X(:,2) = foreach(@(Wx,G)Wx-alpha(1)*G,Wx(:,1),G_saved(:,1));
end
for k = 1:K-1
    Wx(:,k+1) = W.consensus(X(:,k+1));
    [G_saved(:,k+1),F_saved(:,k+1)] = LocalOracles(Fi,X(:,k+1));
    X(:,k+2) = foreach(@(x1, Wx1, x, Wx, G1,G) x1 + Wx1 - 1/2*(x+Wx) - alpha(k+1)*(G1-G), X(:,k+1), Wx(:,k+1), X(:,k), Wx(:,k),G_saved(:,k+1), G_saved(:,k));
end

% (5) Set up the performance measure
switch perf
    case 'Navg_last_it_err' % 1/N sum_i ||x_i^K - x*||2
        metric = 1/N*sumcell(foreach(@(xiK)(xiK-xs)^2,X(:,K+1)));
    case 'Kavg_Navg_it_err' % avg_i avg_k ||x_i^k - x*||2
        metric = 1/((K+1)*N)*sumcell(foreach(@(xiK)(xiK-xs)^2,X(:,:)));
    case 'it_err_last_Navg' % ||avg_i x_i^K - x*||^2
        xperf = sumcell(X(:,K+1))/(N);
        metric = (xperf-xs)^2;
    case 'it_err_Kavg_Navg' % ||avg_i avg_k x_i^k - x*||^2
        xperf = sumcell(X)/((K+1)*N);
        metric = (xperf-xs)^2;
    case 'it_err_last_worst' % max_i ||x_i^K - x*||2
        metric = (X{1,K+1}-xs)^2;
    case 'it_err_last_percentile_worst' % max_{i \in sets 2 and 3} ||x_i^K - x*||2
        wa = round((1-init.percentile)*N)+1;
        metric = (X{wa,K+1}-xs)^2;
        %obj_exclude = 1/(wa-1)*sumcell(foreach(@(xiK)(xiK-xs)^2,X(1:wa-1,K+1)));
        P.AddMultiConstraints(@(xiK)(xiK-xs)^2 >= metric,X(1:wa-1,K+1));
    case 'Navg_last_it_err_combined_with_g' % for rate in DIGing
        [Glast,~] = LocalOracles(Fi,X(:,K+1));
        metric = 1/N*sumcell(foreach(@(xi,gi)(xi-xs)^2 + init.gamma*(gi - 1/N*sumcell(Glast))^2, X(:,K+1), Glast));
    case 'fct_err_last_Navg' % last iterate agent average function error: F(xb(K)) - F(x*)
        xperf = sumcell(X(:,K+1))/(N);
        metric = Fav.value(xperf)-Fs;
    case 'fct_err_last_worst' % worst agent function error: max_i F(xi) - F(x*)
        xperf = X{1,K+1};
        metric = Fav.value(xperf)-Fs;
    case 'fct_err_Kavg_Navg' % average iterate of agent average function error: F(avg_k xb(k)) - F(x*)
        xperf = sumcell(X)/((K+1)*N);
        metric = Fav.value(xperf)-Fs;
    otherwise % default: last_Navg_fct_err
        fprintf("default performance criterion applied: last_Navg_fct_err")
        xperf = sumcell(X(:,K+1))/(N);
        metric = Fav.value(xperf)-Fs;
end

P.PerformanceMetric(metric);

% Activate the trace heuristic for trying to reduce the solution dimension
if trace_Heuristic
    P.TraceHeuristic(1);
end

% (6) Solve the PEP
if verbose
    switch type
        case 'spectral_relaxed'
            fprintf("Spectral PEP formulation for DGD after %d iterations, with %d agents \n",K,N);
            fprintf("Using the following spectral range for the communication matrix: [%1.2f, %1.2f] \n",mat)
        case 'exact'
            fprintf("Exact PEP formulation for DGD after %d iterations, with %d agents \n",K,N);
            fprintf("The used communication matrix is\n")
            disp(mat);
    end
end
out = P.solve(verbose+1);
if verbose, out, end

% (7) Evaluate the output
d = length(double(X{1,1}));

% Evaluating the X and Y solution of PEP.
Xv = zeros(N,K,d); Yv = zeros(N,K, d); grad = zeros(N,K+1,d);
for k = 1:K+1
    for i = 1:N
        if k < K+1
        Yv(i,k,:) = double(Wx{i,k});
        grad(i,k,:) = double(G_saved{i,k});
        end
        Xv(i,k,:) = double(X{i,k});
    end
end
out.X = Xv;
out.Y = Yv;
out.g = grad;
out.xs = double(xs);

% (8) Construct an approximation of the worst communication matrix that links the solutions X and Y
if estimateW
    [Wh.W,Wh.r,Wh.status] = W.estimate(0);
    if verbose && strcmp(type,'spectral_relaxed')
        fprintf("The estimate of the worst matrix is ")
        Wh.W
        eig(Wh.W);
        Wh.r;
        Wh.status;
    end
    out.Wh = Wh;
end

end