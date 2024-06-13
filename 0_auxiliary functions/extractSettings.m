function [n,t,alpha,ATC,type,avg_mat,tv_mat,eq_start,init,perf,fctClass,fctParam,Settings] = extractSettings(S)
% The 'extractSettings' function extracts the settings provided in S, and
% assign default values for those not specified
% INPUT:
%   S: structure with all the settings for the PEP for DGD. 
%   The structure can include the following fields:
%   (unspecified fields will be set to a default value)
%       S.n: number of agents (default = 2)
%       S.t: number of iterations (default = 1)
%       S.alpha: step-size (scalar or vector of t elements) (default = 1)
%       S.avg_mat: averaging matrix description (one of the following options)
%           - (n x n) matrix
%           - the second-largest eigenvalue modulus (scalar)
%           - range of eigenvalues for the averaging matrix
%           (except one): [lb, ub] with -1 < lb <= ub < 1.
%           (default = 0.5)
%       S.tv_mat: boolean, 1 if the averaging matrix can vary across the
%                        iteration and 0 otherwise.
%       S.eq_start: boolean to indicate if the agents start with the same initial iterate
%       S.init: structure with details about the initial conditions
%         init.x: string to specify the initial condition to
%                 consider for the local iterates (x)
%         init.D: real constant to use in the initial condition (cond_x <= D^2)
%         init.grad: string to choose the initial condition to
%                    consider for the local gradients
%         init.E: real constant to use in the initial condition (cond_g <= E^2)
%       S.perf: string to specify the performance criterion to consider in PEP
%       S.fctClass: string to specify the class of functions
%       S.fctParam: structure with the parameter values of the function class

if ~isstruct(S)
    warning("settings should be provided in a single structure - default settings used");
    S = struct();
end

% Number of agents
if isfield(S,'n')
    n = S.n;
else % default
    n = 2;
end

% Number of iterations
if isfield(S,'t')
    t = S.t;
else % default
    t = 1;
end

% step-size(s)
if isfield(S,'alpha')
    if length(S.alpha) == 1 % use the same step-size for each iteration
        alpha = S.alpha*ones(1,t);
    elseif length(S.alpha) == t
        alpha = S.alpha;
    else
        error("the field 'alpha' should be of length 1 (constant step-size) or length t (time-varying step-size)");
    end
else % default
    alpha = 1*ones(1,t);
end

% ATC scheme for the algorithm ?
if isfield(S,'ATC')
    ATC = S.ATC;
else
    ATC = 0;
end

% averaging matrix description
if isfield(S,'avg_mat')
    % (a) Exact formulation (fixed network W)
    if all(size(S.avg_mat) == [n,n])
        type = 'exact';
        avg_mat = S.avg_mat;
    
    % (b) Spectral formulation for symmetric (generalized) doubly stochastic averaging matrices
    elseif length(S.avg_mat)==1 % SLEM description
        type = 'spectral_relaxed'; 
        avg_mat = [-S.avg_mat,S.avg_mat];
    elseif length(S.avg_mat)==2 % eigenvalue range
        type = 'spectral_relaxed'; 
        avg_mat = S.avg_mat;    
    else
        error("the field 'avg_mat' should be a matrix of size n x n (fixed averaging matrix), or a vector of length 1 (SLEM), or length 2 (range of eigenvalues)");
    end
elseif isfield(S,'lam') % alternative notation
    type = 'spectral_relaxed';
    if length(S.avg_mat)==1 % SLEM description
        avg_mat = [-S.lam,S.lam];
    elseif length(S.avg_mat)==2 % eigenvalue range
        avg_mat = S.lam;
    end
else % default
    type = 'spectral_relaxed';
    avg_mat = [-0.5,0.5]; % SLEM
end

% is the averaging matrix time-varying ?
if isfield(S,'tv_mat')
    tv_mat = S.tv_mat;
else % default
    tv_mat = 0;
end

% equal start for all the agents
if isfield(S,'eq_start')
    eq_start = S.eq_start;
else % default
    eq_start = 0;
end

% class of local functions
if isfield(S,'fctClass')
    fctClass = convertStringsToChars(S.fctClass);
else % default
    fctClass = 'SmoothStronglyConvex';
end

% parameters for the class of functions
if isfield(S,'fctParam')
    fctParam = S.fctParam;
else % default
    if strcmp(fctClass, 'SmoothStronglyConvex')
        fctParam.L = 1;
        fctParam.mu = 0.1;
    elseif strcmp(fctClass,'ConvexBoundedGradient')
        fctParam.R = 1;
    else
        fctParam.L=Inf;
        fctParam.mu=0;
        fctParam.D=Inf;
        fctParam.R=Inf;
    end
end

% initial conditions
if isfield(S,'init')
    init = lower(S.init);
    if ~isfield(S.init,'D')
        init.D = 1;
    end
    if ~isfield(S.init,'E')
        init.E = 1;
    end
    if ~isfield(S.init,'gamma')
        init.gamma = alpha(1);
    end
    if ~isfield(S.init,'x')
        init.x = '';
    end
    if ~isfield(S.init,'grad')
        init.grad = '';
    end
else % default
    init.x = 'bounded_navg_it_err';
    init.grad = '';
    init.D = 1;
    init.E = 1;
end

% Performance Criterion
if isfield(S,'perf')
    perf = lower(S.perf);
else % default
    perf = 'fct_err_last_navg';
end

% Summary of the Settings
Settings.n = n; Settings.t = t; Settings.alpha = alpha; Settings.ATC = ATC;
Settings.type = type; Settings.avg_mat = avg_mat; Settings.tv_mat=tv_mat;
Settings.eq_start = eq_start; Settings.init= init; Settings.perf=perf;
Settings.fctClass = fctClass; Settings.fctParam = fctParam;

end