% In this example, we consider K iterations of the decentralized subgradient
% descent (DGD) with N agents, that each holds a local convex function Fi 
% with bounded subgradients, for solving the following decentralized problem:
%   min_x F(x);     where F(x) is the average of local functions Fi. 
% For notational convenience we denote xs=argmin_x F(x), and Fs = F(xs). 
% Agents start all with the same iterate x0 such that ||x0 - xs||^2 <= 1. 
%
% This example shows how to obtain the worst-case performance of DGD with
% PESTO in that case. The performance criterion used is F(xav) - Fs, where
% xav is the average of the iterates over all the agents and all the iterations.
% Communications between the agents are represented with averaging consensus steps, 
% that can be formulated in two different ways in the PEP problem, leading to 
% different types of worst-case solution:
%   (a) Using a fixed averaging matrix, which leads to exact worst-case results 
%   that are specific to the choosen matrix.
%   (b) Using an entire spectral class of averaging matrices: 
%   the symmetric (generealized) doubly stochastic matrices with a given range of eigenvalues
%   This leads to a relaxation of PEP, providing worst-case valid for any 
%   matrix of the spectral class.
% Both formulations can be tested here but (b) is used by default.

% For details, see
%   [1] Colla, Sebastien, and Julien M. Hendrickx. "Automated Worst-Case
%   Performance Analysis of Decentralized Gradient Descent" (2021).

verbose = 1;                % Print the problem set up and the results
trace_heuristic = 0;        % heuristic to minimize the dimension of the worst-case (1 to activate)

%%% Set up general problem parameters %%%
% The system
N = 3;                      % Number of agents

% (a) Exact formulation (fixed network W)
% Uncomment to use it in place of (b) Spectral formulation
%type = 'exact';
%mat = [0,0.5,0.5;0.5,0,0.5;0.5,0.5,0]; 

% (b) Spectral formulation
% Comment if you use (a) Exact formulation
type = 'spectral_relaxed';  % type of representation for the averaging matrix
mat = [-0.5,0.5];           % Range of eigenvalues for the symmetric(generalized) doubly stochastic averaging matrix W

% The algorithm
K = 10;                     % Number of iterations of DGD
alpha = 1/sqrt(K);          % Step-size used in DGD (constant)
%alpha = 1./(1:K);          % Alternative: Step-sizes used in DGD (diminishing)
equalStart = 1;             % All agents starts with the same iterate x0
D = 1;                      % Constant for the initial condition: ||x0 - xs||^2 <= D^2
time_varying_mat = 0;       % The same averaging matrix is used at each iteration (if 1, no constraints for imposing that the same matrix is used at each iteration)

% (0) Initialize an empty PEP
P = pep();   

% (1) Set up the local and global objective functions
fctClass = 'ConvexBoundedGradient'; % Class of functions to consider for the worst-case
fctParam.R = 1;                     % Bounded Gradient constant ||g||^2 <= R^2.
returnOpt = 0;
[Fi,Fav,~,~] = P.DeclareMultiFunctions(fctClass,fctParam,N,returnOpt);
[xs,Fs] = Fav.OptimalPoint();       % optimum of the global function

% Iterates cells
X = cell(N, K+1);                   % local iterates
Y = cell(N, K);                     % Y = WX
F_saved = cell(N,K+1);
G_saved = cell(N,K+1);

% (2) Set up the starting points and initial conditions
X(:,1) = P.MultiStartingPoints(N,equalStart);
[G_saved(:,1),F_saved(:,1)] = LocalOracles(Fi,X(:,1)); % F_1(X(1,1)), F_2(X(2,1)), ...
P.AddMultiConstraints(@(xi) (xi-xs)^2 <= D^2, X(:,1)); %initial condition: ||xi0 - xs||^2 <= D^2

% (3) Set up the averaging matrix
W = P.DeclareConsensusMatrix(type,mat,time_varying_mat);

% (4) Algorithm (DGD)
% Step-size 
if length(alpha) == 1 % use the same step-size for each iteration
    alpha = alpha*ones(1,K);
end

% Iterations
for k = 1:K    
    Y(:,k) = W.consensus(X(:,k));                                       % Consensus step: Y = WX
    X(:,k+1) = foreach(@(y,G) y-alpha(k)*G, Y(:,k), G_saved(:,k));      % Gradient step
    %          for each agent: (expression for the update, cells of variables to input in the expression) 
    [G_saved(:,k+1),F_saved(:,k+1)] = LocalOracles(Fi,X(:,k+1));        % Eval F and G at k+1 (for all agents)
    % not always needed for the points at K+1 (depending on the perf measure).
end

% (5) Set up the performance measure
xav = sumcell(X)/((K+1)*N);             % Average over all agents, all iterations
P.PerformanceMetric(Fav.value(xav)-Fs); % Worst-case evaluated as F(xav)-F(x*)
%P.PerformanceMetric((xav-xs)^2);

% Alternative: Average only for the last iteration
%xavLast = sumcell(X(:,K+1))/N; 
%P.PerformanceMetric(Fav.value(xavLast)-Fs); % OR P.PerformanceMetric((xavLast - xs)^2);

% Activate the trace heuristic for trying to reduce the solution dimension
P.TraceHeuristic(trace_heuristic);

% (6) Solve the PEP
if verbose
    switch type
        case 'spectral_relaxed'
            fprintf("Spectral PEP formulation for DGD after %d iterations, with %d agents \n",K,N);
            fprintf("Using the following spectral range for the averaging matrix: [%1.2f, %1.2f] \n",mat)
        case 'exact'
            fprintf("Exact PEP formulation for DGD after %d iterations, with %d agents \n",K,N);
            fprintf("The used averaging matrix is\n")
            disp(mat);
    end
end
out = P.solve(verbose+1);
if verbose, out, end

% (7) Evaluate the output
wc = out.WCperformance;

% (8) Construct an approximation of the worst averaging matrix that links the solutions X and Y
[Wh.W,Wh.r,Wh.status] = W.estimate(0);
if verbose && strcmp(type,'spectral_relaxed')
    fprintf("The estimate of the worst matrix is ")
    Wh.W
    eig(Wh.W);
    Wh.r;
    Wh.status;
end


% (9) Comparison with theoretical guarantee [1, Theorem 5]
% This guarantee is valid for F(xav)-F(x*) and equalStart = 1.
% It always applies to a spectral class of averaging matrices.
switch type
    case 'spectral_relaxed'
        lam2 = max(abs(mat));
    case 'exact'
        lam2 = max(abs(eig(mat-1/N*ones(N,N))));
end
wc_theo = (D^2 + fctParam.R^2)./(2*sqrt(K)) + 2*fctParam.R^2./(sqrt(K)*(1-lam2));

if verbose
    fprintf("Performance guarantee from PESTO: %1.4f \n",wc);
    fprintf("Theoretical guarantee from [1]: %1.4f\n\n",wc_theo);
end