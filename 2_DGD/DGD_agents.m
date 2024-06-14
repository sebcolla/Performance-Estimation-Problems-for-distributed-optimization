function out = DGD_agents(Settings)
% Compute the worst-case performance of DGD [1] under the 'Settings' provided
% INPUT:
%   Settings: structure with all the settings to use in the PEP for DGD. 
%   The structure can include the following fields:
%   (unspecified fields will be set to a default value)
%       Settings.n: number of agents (default = 2)
%       Settings.t: number of iterations (default = 1)
%       Settings.alpha: step-size (scalar or vector of t elements) (default = 1)
%       Settings.avg_mat: averaging matrix description (one of the following options)
%           - (n x n) matrix
%           - the second-largest eigenvalue modulus (scalar)
%           - range of eigenvalues for the averaging matrix
%           (except one): [lb, ub] with -1 < lb <= ub < 1.
%           (default = 0.5)
%       Settings.tv_mat: boolean, 1 if the averaging matrix can vary across the
%                        iteration and 0 otherwise. (default = 0)
%       Settings.eq_start: boolean to indicate if the agents start with the
%       same initial iterate (default = 0)
%       Settings.init: structure with details about the initial conditions
%                init.x:    string to specify the initial condition to consider for 
%                           the local iterates (x) (default = 'bounded_navg_it_err')
%                init.D:    real constant to use in the initial condition (cond_x <= D^2) (default = 1)
%                init.grad: string to choose the initial condition to consider for 
%                           the local gradients (default = None)
%                init.E:    real constant to use in the initial condition (cond_g <= E^2) (default = 1)
%                init.gamma: real coefficient to use in combined conditions (cond_x + gamma*cond_g <= D^2)
%                           (default = 1)
%       Settings.perf: string to specify the performance criterion to consider in PEP
%                      (default = 'fct_err_last_navg')
%       Settings.fctClass: string to specify the class of functions
%                          (default = 'SmoothStronglyConvex')
%       Settings.fctParam: structure with the parameter values of the function class
%
% OUTPUT: structure with details about the worst-case solution of the PEP
%   solverDetails: structure with solver details
%   WCperformance: worst-case performance value
%   G:  Gram matrix solution of the PEP
%   dualvalues_LMIs: dual values of the LMI PEP constraints
%   dualnames_LMIs:  coresponding names of the LMI contraints
%   dualvalues:      dual values of the scalar constraints
%   dualnames:       coresponding names of the contraints
%   Settings:        structure with all the settings used in the PEP
%                   (including all the default values that have been set)
%   Possible additional fields:
%       iterates (X) and gradients (g) if 'eval_out = 1' in the code
%       the worst-case averaging matrix (Wh) if 'estim_W = 1' in the code 
%
% Reference
%   [1] Angelia Nedic and Asuman Ozdaglar. Distributed subgradient methods 
%   for multi-agent optimization. IEEE Transactions on Automatic Control, 2009.


verbose = 0;            % print the problem set up and the results
trace_heuristic = 0;    % heuristic to minimize the dimension of the worst-case (1 to activate)
eval_out = 0;           % evaluate the worst-case local iterates and gradients and add them to the output
estim_W = 0;            % estimate the worst-case averaging matrix

%%% Set up performance estimation settings %%%
if nargin == 1
    Settings = extractSettings(Settings);
else
    warning("settings should be provided in a single structure - default settings used")
    Settings = extractSettings(struct());
end
n=Settings.n; t=Settings.t; alpha=Settings.alpha; 
type=Settings.type; mat=Settings.avg_mat; tv_mat=Settings.tv_mat; 
eq_start=Settings.eq_start; init=Settings.init; perf=Settings.perf; 
fctClass=Settings.fctClass; fctParam=Settings.fctParam;

if verbose
    fprintf("Settings provided for the PEP:\n");
    fprintf("n=%d, t=%d, alpha=%1.2f, type=%s, tv_mat=%d, eq_start=%d,\ninit_x=%s, init_grad=%s, perf=%s, fctClass=%s,\n",n,t,alpha(1),type,tv_mat,eq_start,init.x,init.grad,perf,fctClass);
    fprintf('avg_mat = ['); fprintf('%g ', mat); fprintf(']\n');
    fprintf("------------------------------------------------------------------------------------------\n");
end

% (0) Initialize an empty PEP
P = pep();

% (1) Set up the local and global objective functions
returnOpt = 0; % or 1 if you need the optimal point of each local function
[Fi,Fav,~,~] = P.DeclareMultiFunctions(fctClass,fctParam,n,returnOpt);
[xs,Fs] = Fav.OptimalPoint();

% Iterates cells
X = cell(n, t+1);       % local iterates
F_saved = cell(n,t+1);  % local function values
G_saved = cell(n,t+1);  % local gradient vectors

% (2) Set up the starting points and initial conditions
X(:,1) = P.MultiStartingPoints(n,eq_start);
[G_saved(:,1),F_saved(:,1)] = LocalOracles(Fi,X(:,1));

% Initial condition for x0
switch init.x
    case {'bounded_navg_it_err','0'}      % avg_i ||xi0 - xs||^2 <= D^2
        P.AddConstraint(1/n*sumcell(foreach(@(xi) (xi-xs)^2,X(:,1))) <= init.D^2);
    case {'uniform_bounded_it_err','1'}   %||xi0 - xs||^2 <= D^2 for all i
        P.AddMultiConstraints(@(xi) (xi-xs)^2 <= init.D^2, X(:,1));
    case {'navg_it_err_combined_grad','2'}   % (avg_i ||xi0 - xs||^2) + gamma* (avg_i ||gi0 - avg_i(gi0)||^2) <= D^2
        metric = 1/n*sumcell(foreach(@(x0, g0)(x0-xs)^2 + init.gamma*(g0 - 1/n*sumcell(G_saved(:,1)))^2,X(:,1), G_saved(:,1)));
        P.AddConstraint(metric <= init.D^2);
    otherwise % default is bounded_navg_it_err
        P.AddConstraint(1/n*sumcell(foreach(@(xi) (xi-xs)^2,X(:,1))) <= init.D^2);
end

% Initial condition for g0
switch init.grad
    case {'bounded_navg_grad0','0'}       % avg_i ||gi0||^2 <= E^2
        P.AddConstraint(1/n*sumcell(foreach(@(gi) gi^2, G_saved(:,1))) <= init.E^2);
    case {'uniform_bounded_grad0','1'}    % ||gi0||^2 <= E^2 for all i
        P.AddMultiConstraints(@(gi) gi^2 <= init.E^2, G_saved(:,1));
    case {'bounded_grad0_cons_err','2'}    % avg_i ||gi0 - avg_i(gi0)||^2 <= E^2
        P.AddConstraint(1/n*sumcell(foreach(@(gi) (gi - 1/n*sumcell(G_saved(:,1)))^2,G_saved(:,1))) <= init.E^2);
    case {'bounded_navg_grad*','3'}       % avg_i ||gi(x*)||^2 <= E^2
        [gis,~] = LocalOracles(Fi,repmat({xs},n,1));
        P.AddConstraint(1/n*sumcell(foreach(@(gi) gi^2, gis)) <= init.E^2);
    case {'uniform_bounded_grad*','4'}    % ||gi(x*)||^2 <= E^2 for all i
        [gis,~] = LocalOracles(Fi,repmat({xs},n,1));
        P.AddMultiConstraints(@(gi) gi^2 <= init.E^2, gis);
    otherwise % default
        % none
end

% (3) Set up the averaging matrix
W = P.DeclareConsensusMatrix(type,mat,tv_mat);

% (4) Algorithm (DGD)
% Iterations
for k = 1:t
    % x(k+1) = W x(k) - alpha*g(k)
    X(:,k+1) = foreach(@(y,G) y-alpha(k)*G, W.consensus(X(:,k)), G_saved(:,k)); % update for each agent
    %          foreach(expression for the update, cells of variables to input in the expression)
    [G_saved(:,k+1),F_saved(:,k+1)] = LocalOracles(Fi,X(:,k+1));        % Eval fi and gi at k+1 (for all agents)
    % not always needed for the points at t+1 (depending on the perf measure).
end

% (5) Set up the performance measure
switch perf
    case {'navg_last_it_err','0'} % 1/n sum_i ||x_i^t - x*||2
        metric = 1/n*sumcell(foreach(@(xit)(xit-xs)^2,X(:,t+1)));
    case {'tavg_navg_it_err','1'} % avg_i avg_k ||x_i^k - x*||2
        metric = 1/((t+1)*n)*sumcell(foreach(@(xit)(xit-xs)^2,X(:,:)));
    case {'navg_it_err_combined_grad','2'}
        metric = 1/n*sumcell(foreach(@(xi,si)(xi-xs)^2 + init.gamma*(si - 1/n*sumcell(G_saved(:,t+1)))^2, X(:,t+1), G_saved(:,t+1)));
    case {'it_err_last_navg','3'} % ||avg_i x_i^t - x*||^2
        xperf = sumcell(X(:,t+1))/(n);
        metric = (xperf-xs)^2;
    case {'it_err_tavg_navg','4'} % ||avg_i avg_k x_i^k - x*||^2
        xperf = sumcell(X)/((t+1)*n);
        metric = (xperf-xs)^2;
    case {'it_err_last_worst','5'} % max_i ||x_i^t - x*||2
        metric = (X{1,t+1}-xs)^2;
    case {'fct_err_last_navg','6'} % last iterate agent average function error: F(xb(t)) - F(x*)
        xperf = sumcell(X(:,t+1))/(n);
        metric = Fav.value(xperf)-Fs;
    case {'fct_err_last_worst','7'} % worst agent function error: max_i F(xi) - F(x*)
        xperf = X{1,t+1};
        metric = Fav.value(xperf)-Fs;
    case {'fct_err_tavg_navg','8'} % average iterate of agent average function error: F(avg_k xb(k)) - F(x*)
        xperf = sumcell(X)/((t+1)*n);
        metric = Fav.value(xperf)-Fs;
    otherwise % default: last_navg_fct_err
        xperf = sumcell(X(:,t+1))/(n);
        metric = Fav.value(xperf)-Fs;
end

P.PerformanceMetric(metric);

% Activate the trace heuristic for trying to reduce the solution dimension
P.TraceHeuristic(trace_heuristic);

% (6) Solve the PEP
if verbose
    switch type
        case 'spectral_relaxed'
            fprintf("Spectral PEP formulation for DGD after %d iterations, with %d agents \n",t,n);
            fprintf("Using the following spectral range for the averaging matrix: [%1.2f, %1.2f] \n",mat)
        case 'exact'
            fprintf("Exact PEP formulation for DGD after %d iterations, with %d agents \n",t,n);
            fprintf("The used averaging matrix is\n")
            disp(mat);
    end
end
out = P.solve(verbose+1);
out.Settings = Settings;

% (7) Evaluate the output
if eval_out
    d = length(double(X{1,1}));
    % Evaluating the X and Y solution of PEP.
    Xv = zeros(n,t,d); grad = zeros(n,t+1,d);
    for k = 1:t+1
        for i = 1:n
            Xv(i,k,:) = double(X{i,k});
            grad(i,k,:) = double(G_saved{i,k});
        end
    end
    out.X = Xv;
    out.g = grad;
    out.xs = double(xs);
end

% (8) (Try to) Recover the worst-case averaging matrix that links the solutions X and Y
if estim_W
    [Wh.W,Wh.r,Wh.status] = W.estimate(0);
    if verbose && strcmp(type,'spectral_relaxed')
        fprintf("The estimate of the worst-case averaging matrix is ")
        Wh.W
        %other info that could be printed : eig(Wh.W); Wh.r; Wh.status;
    end
    out.Wh = Wh;
end

if verbose
    out
    fprintf("--------------------------------------------------------------------------------------------\n");
    switch type
        case 'spectral_relaxed'
            fprintf("Performance guarantee obtained with PESTO: %1.2f  (valid for any symmetric doubly stochastic matrix such that |lam_2|<=%1.1f)\n",out.WCperformance, max(abs(mat)));
        case 'exact'
            fprintf("Performance guarantee obtained with PESTO: %1.2f  (only valid for the specific matrix W)\n",out.WCperformance);
    end
end
end