function out = DGD_agents(Settings)
% Compute the worst-case performance of t steps of DGD
% INPUT:
%   n: number of agents
%   t: number of iterations
%   alpha: step-size (scalar or vector of t elements)
%   avg_mat: averaging matrix description (one of the following options)
%           - (n x n) matrix
%           - the second-largest eigenvalue modulus (scalar)
%           - range of eigenvalues for the averaging matrix
%           (except one): [lb, ub] with -1 < lb <= ub < 1.
% possible additional inputs
%   tv_mat: boolean, 1 if the averaging matrix can vary across the
%   iteration and 0 otherwise.
%   eq_start: boolean to indicate if the agents start with the same initial iterate
%   init: string to choose the initial condition to consider in PEP
%   perf: string to choose the performance criterion to consider in PEP
%   fctClass:
%   fctParam:
% OUTPUT: structur with a lot of information about the worst-case solution
% of the PEP
%

verbose = 1;            % print the problem set up and the results
trace_heuristic = 0;    % heuristic to minimize the dimension of the worst-case (1 to activate)
eval_out = 0;           % evaluate the worst-case local iterates and gradients and add them to the output
estim_W = 0;            % estimate the worst-case averaging matrix

%%% Set up performance estimation settings %%%
if nargin == 1
    [n,t,alpha,type,mat,tv_mat,eq_start,init,perf,fctClass,fctParam] = extractSettings(Settings);
else
    warning("settings should be provided in a single structure - default settings used")
    [n,t,alpha,type,mat,tv_mat,eq_start,init,perf,fctClass,fctParam] = extractSettings(struct());
end
if verbose
    fprintf("Settings provided for the PEP:\n");
    fprintf("n=%d, t=%d, alpha=%1.2f, type=%s, tv_mat=%d, eq_start=%d,\ninit=%s, perf=%s, fctClass=%s,\n",n,t,alpha,type,tv_mat,eq_start,init.type,perf,fctClass);
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
Y = cell(n, t);         % Y = WX
F_saved = cell(n,t+1);  % local function values
G_saved = cell(n,t+1);  % local gradient vectors

% (2) Set up the starting points and initial conditions
X(:,1) = P.MultiStartingPoints(n,eq_start);
[G_saved(:,1),F_saved(:,1)] = LocalOracles(Fi,X(:,1));


switch init.type
    case 'bounded_navg_it_err' % avg_i ||xi0 - xs||^2 <= D^2
        P.AddConstraint(1/n*sumcell(foreach(@(xi) (xi-xs)^2,X(:,1))) <= init.D^2);
        %P.AddConstraint(1/n*sumcell(foreach(@(gi) (gi - 1/n*sumcell(G_saved(:,1)))^2,G_saved(:,1))) <= 1); % avg_i ||gi0 - avg_i(gi0)||^2 <= E^2
        %[Gis,~] = LocalOracles(Fi,repmat({xs},n,1));
        %P.AddMultiConstraints(@(gi) gi^2 <= 1, Gis);
    case 'uniform_bounded_it_err' %initial condition: ||xi0 - xs||^2 <= D^2
        P.AddMultiConstraints(@(xi) (xi-xs)^2 <= init.D^2, X(:,1));
    otherwise % default is bounded_avg_it_err
        P.AddConstraint(1/n*sumcell(foreach(@(xi) (xi-xs)^2,X(:,1))) <= init.D^2); % avg_i ||xi0 - xs||^2 <= D^2
end

% case 'diging_like'
%         P.AddConstraint(1/N*sumcell(foreach(@(xi) (xi-xs)^2,X(:,1))) <= D^2);   % avg_i ||xi0 - xs||^2 <= D^2
%         P.AddConstraint(1/N*sumcell(foreach(@(gi) (gi - 1/N*sumcell(G_saved(:,1)))^2,G_saved(:,1))) <= E^2); % avg_i ||gi0 - avg_i(gi0)||^2 <= E^2
%     case 'diging_like_combined'
%         metric = 1/N*sumcell(foreach(@(x0, g0)(x0-xs)^2 + init.gamma*(g0 - 1/N*sumcell(G_saved(:,1)))^2,X(:,1), G_saved(:,1)));
%         P.AddConstraint(metric <= D^2);
%     case 'uniform_bounded_iterr_local_grad0'
%         P.AddMultiConstraints(@(xi) (xi-xs)^2 <= D^2, X(:,1));                  % ||xi0 - xs||^2 <= D^2 for all i
%         P.AddMultiConstraints(@(gi) gi^2 <= E^2, G_saved(:,1));                 % ||gi0||^2 <= E^2 for all i
%     case 'bounded_navg_iterr_local_grad0'
%         P.AddConstraint(1/N*sumcell(foreach(@(xi) (xi-xs)^2,X(:,1))) <= D^2);   % avg_i ||xi0 - xs||^2 <= D^2
%         P.AddConstraint(1/N*sumcell(foreach(@(gi) gi^2, G_saved(:,1))) <= E^2); % avg_i ||gi0||^2 <= E^2
%     case 'uniform_bounded_iterr_local_grad*'
%         P.AddMultiConstraints(@(xi) (xi-xs)^2 <= D^2, X(:,1));                  % ||xi0 - xs||^2 <= D^2 for all i
%         [Gis,~] = LocalOracles(Fi,repmat({xs},N,1));
%         P.AddMultiConstraints(@(gi) gi^2 <= E^2, Gis);                          % ||gi(x*)||^2 <= E^2 for all i
%     case 'bounded_avg_iterr_local_grad*'
%         P.AddConstraint(1/N*sumcell(foreach(@(xi) (xi-xs)^2,X(:,1))) <= D^2);   % avg_i ||xi0 - xs||^2 <= D^2
%         [Gis,~] = LocalOracles(Fi,repmat({xs},N,1));
%         P.AddConstraint(1/N*sumcell(foreach(@(gi) gi^2, Gis)) <= E^2);          % avg_i ||gi(x*)||^2 <= E^2
%     otherwise %'uniform_bounded_iterr_local_grad*'
%         P.AddConstraint(1/N*sumcell(foreach(@(xi) (xi-xs)^2,X(:,1))) <= D^2);   % avg_i ||xi0 - xs||^2 <= D^2
%         [Gis,~] = LocalOracles(Fi,repmat({xs},N,1));
%         P.AddMultiConstraints(@(gi) gi^2 <= E^2, Gis);

% (3) Set up the averaging matrix
W = P.DeclareConsensusMatrix(type,mat,tv_mat);

% (4) Algorithm (DGD)
% Iterations
for k = 1:t
    Y(:,k) = W.consensus(X(:,k));                                       % Consensus step
    X(:,k+1) = foreach(@(y,G) y-alpha(k)*G, Y(:,k), G_saved(:,k));      % Gradient step
    %          for each agent: (expression for the update, variables to input in the expression)
    [G_saved(:,k+1),F_saved(:,k+1)] = LocalOracles(Fi,X(:,k+1));        % Eval F and G at k+1 (for all agents)
    % not always needed for the points at t+1 (depending on the perf measure).
end

% (5) Set up the performance measure
switch perf
    case {'navg_last_it_err','0'} % 1/n sum_i ||x_i^t - x*||2
        metric = 1/n*sumcell(foreach(@(xit)(xit-xs)^2,X(:,t+1)));
    case {'tavg_navg_it_err','1'} % avg_i avg_k ||x_i^k - x*||2
        metric = 1/((t+1)*n)*sumcell(foreach(@(xit)(xit-xs)^2,X(:,:)));
    case {'it_err_last_navg','2'} % ||avg_i x_i^t - x*||^2
        xperf = sumcell(X(:,t+1))/(n);
        metric = (xperf-xs)^2;
    case {'it_err_tavg_navg','3'} % ||avg_i avg_k x_i^k - x*||^2
        xperf = sumcell(X)/((t+1)*n);
        metric = (xperf-xs)^2;
    case {'it_err_last_worst','4'} % max_i ||x_i^t - x*||2
        metric = (X{1,t+1}-xs)^2;
    case {'fct_err_last_navg','5'} % last iterate agent average function error: F(xb(t)) - F(x*)
        xperf = sumcell(X(:,t+1))/(n);
        metric = Fav.value(xperf)-Fs;
    case {'fct_err_last_worst','6'} % worst agent function error: max_i F(xi) - F(x*)
        xperf = X{1,t+1};
        metric = Fav.value(xperf)-Fs;
    case {'fct_err_tavg_navg','7'} % average iterate of agent average function error: F(avg_k xb(k)) - F(x*)
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
if verbose, out, end

% (7) Evaluate the output
if eval_out || estim_W
    d = length(double(X{1,1}));
    % Evaluating the X and Y solution of PEP.
    Xv = zeros(n,t,d); Yv = zeros(n,t, d); grad = zeros(n,t+1,d);
    for k = 1:t+1
        for i = 1:n
            if k < t+1
                Yv(i,k,:) = double(Y{i,k});
            end
            Xv(i,k,:) = double(X{i,k});
            grad(i,k,:) = double(G_saved{i,k});
        end
    end
    out.X = Xv;
    out.Y = Yv;
    out.g = grad;
    out.xs = double(xs);
end

% (8) Construct an approximation of the worst averaging matrix that links the solutions X and Y
if estim_W
    [Wh.W,Wh.r,Wh.status] = W.estimate(0);
    if verbose && strcmp(type,'spectral_relaxed')
        fprintf("The estimate of the worst matrix is ")
        Wh.W
        %other info that could be printed : eig(Wh.W); Wh.r; Wh.status;
    end
    out.Wh = Wh;
end

if verbose
    fprintf("--------------------------------------------------------------------------------------------\n");
    switch type
        case 'spectral_relaxed'
            fprintf("Performance guarantee obtained with PESTO: %1.2f  (valid for any symmetric doubly stochastic matrix such that |lam_2|<=%1.1f)\n",out.WCperformance, max(abs(mat)));
        case 'exact'
            fprintf("Performance guarantee obtained with PESTO: %1.2f  (only valid for the specific matrix W)\n",out.WCperformance);
    end
end
end