% In this example, we consider 1 iterations of the DIGing algorithm [1]
% with N agents, that each holds a local L-smooth mu-strongly convex
% function Fi, for solving the following decentralized problem:
%   min_x F(x);     where F(x) is the average of local functions Fi.
% For notational convenience we denote xs=argmin_x F(x).
%
% This example shows how to obtain the worst-case convergence rate for DIGing with PESTO.
% It is based on the following performance metric that is used both for the initial
% condition and for the performance criterion we maximize:
%   avg_i( ||xi - xs||^2 + gamma*||si - avg_i(grad_fi(xi))||^2 )
% Another initial condition, preserved by DIGing is also needed:
%   sum_i si0 = sum_i grad Fi(xi0)
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
%   [1] A. Nedic, A. Olshevsky, and W. Shi, “Achieving geometric convergence
%   for distributed optimization over time-varying graphs,” SIAM Journal on
%   Optimization, 2016.
%   [2] Colla, Sebastien, and Julien M. Hendrickx. "Automatic Performance Estimation
%   for Decentralized Optimization" (2022).

verbose = 1;                % print the problem set up and the results
trace_heuristic = 0;        % heuristic to minimize the dimension of the worst-case (1 to activate)

%%% Set up general problem parameters %%%
% The system
N = 2;                      % Number of agents

% (a) Exact formulation (fixed network W)
% Uncomment to use it in place of (b) Spectral formulation
% type = 'exact';
% mat = [0.25,0.75;0.75,0.25]; 
% lam = max(abs(eig(mat-1/N*ones(N,N))));

% (b) Spectral formulation
% Comment if you use (a) Exact formulation
type = 'spectral_relaxed';  % type of representation for the averaging matrix
lam = 0.75;
mat = [-lam,lam];           % Range of eigenvalues for the symmetric(generalized) doubly stochastic averaging matrix W

% The algorithm
alpha = 0.44*(1-lam)^2;     % Step-size used in DIGing (constant) (hand-tuned formula)
equalStart = 0;             % initial iterates are not necessarily equal for each agent
M = 1;                      % Constants for the initial conditions
time_varying_mat = 0;       % no impact when only 1 iteration

% (0) Initialize an empty PEP
P = pep();   

% (1) Set up the local and global objective functions
fctClass = 'SmoothStronglyConvex'; % Class of functions to consider for the worst-case
fctParam.L  = 1;
fctParam.mu = 0.1;
returnOpt = 0;
[Fi,Fav,~,~] = P.DeclareMultiFunctions(fctClass,fctParam,N,returnOpt);
[xs,~] = Fav.OptimalPoint(); 

% Iterates cells
X = cell(N, 2);           % local iterates
S = cell(N, 2);             % S contains the local estimates of the global gradient
F_saved = cell(N,2);
G_saved = cell(N,2);

% (2) Set up the starting points and initial conditions
X(:,1) = P.MultiStartingPoints(N,equalStart);
S(:,1) = P.MultiStartingPoints(N,equalStart);
[G_saved(:,1),F_saved(:,1)] = LocalOracles(Fi,X(:,1));
P.AddConstraint((sumcell(S(:,1)) - sumcell(G_saved(:,1)))^2 == 0); % sum_i si0 = sum_i grad Fi(xi0) 
                                                                   % remain valid for any other iteration (check DIGing updates)
gamma = alpha/fctParam.L;
metric0 = 1/N*sumcell(foreach(@(x0, s0)(x0-xs)^2 + gamma*(s0 - 1/N*sumcell(G_saved(:,1)))^2, X(:,1), S(:,1)));
          % avg_i( ||xi0 - xs||^2 + gamma*||si0 - avg_i(grad_fi(xi0))||^2 )
P.AddConstraint(metric0 <= M);

% (3) Set up the averaging matrix
W = P.DeclareConsensusMatrix(type,mat,time_varying_mat);

% (4) Algorithm (DIGing)
% For 1 iteration only
X(:,2) = foreach(@(Wx,S)Wx-alpha*S,W.consensus(X(:,1)),S(:,1));
[G_saved(:,2),F_saved(:,2)] = LocalOracles(Fi,X(:,2));
S(:,2) = foreach(@(Ws,G2,G1) Ws + G2-G1, W.consensus(S(:,1)), G_saved(:,2), G_saved(:,1)); 

% (5) Set up the performance measure
metric = 1/N*sumcell(foreach(@(xi, si)(xi-xs)^2 + gamma*(si - 1/N*sumcell(G_saved(:,2)))^2,X(:,2), S(:,2)));
        % avg_i( ||xi1 - xs||^2 + gamma*||si1 - avg_i(grad_fi(xi1))||^2 )

P.PerformanceMetric(metric); 

% Activate the trace heuristic for trying to reduce the solution dimension
P.TraceHeuristic(trace_heuristic);

% (6) Solve the PEP
if verbose
    switch type
        case 'spectral_relaxed'
            fprintf("Spectral PEP formulation for 1 iteration of DIGing, with %d agents \n",N);
            fprintf("Using the following spectral range for the averaging matrix: [%1.2f, %1.2f] \n",mat)
        case 'exact'
            fprintf("Exact PEP formulation for 1 iteration of DIGing, with %d agents \n",N);
            fprintf("The used averaging matrix is\n")
            disp(mat);
    end
end

out = P.solve(verbose+1);
if verbose, out, end

% (7) Evaluate the output
wc = out.WCperformance;

% (8) Construct an approximation of the worst averaging matrix used
[Wh.W,Wh.r,Wh.status] = W.estimate(0);

% (9) Comparison with theoretical guarantee [1, Theorem 3.14]
% This guarantee always applies to a spectral class of averaging matrices.
wc_theo = max(sqrt(1-alpha*fctParam.mu/1.5),(sqrt(alpha*2*fctParam.L*(1+4*sqrt(N)*sqrt(fctParam.L/fctParam.mu))) + lam));
msg_theo = 'none in the given setting';
if wc_theo <= 1
    msg_theo = sprintf('%1.5f', wc_theo);
end
if verbose
    fprintf("Performance guarantee from PESTO: %1.5f \n",wc);
    fprintf("Theoretical guarantee from [1]: %s \n\n",msg_theo);
end

