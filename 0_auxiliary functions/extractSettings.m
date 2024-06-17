function set = extractSettings(S)
% The 'extractSettings' function extracts the settings provided in S, and
% assign default values for those not specified
% INPUT:
%   S: structure with all the settings to use in PEP. 
%   The structure can include the following fields:
%   (unspecified fields will be set to a default value)
%       S.n: number of agents (default = 2)
%       S.t: number of iterations (default = 1)
%       S.alpha: step-size (scalar or vector of t elements) (default = 1)
%       S.ATC: boolean to indicate if ATC scheme of the algorithm should be used (1) or not (0)
%                     (default = 0)  
%       S.avg_mat: averaging matrix description (one of the following options)
%           - (n x n) matrix
%           - the second-largest eigenvalue modulus (scalar)
%           - range of eigenvalues for the averaging matrix
%           (except one): [lb, ub] with -1 < lb <= ub < 1.
%           (default = 0.5)
%       S.tv_mat: boolean, 1 if the averaging matrix can vary across the
%                        iteration and 0 otherwise. (default = 0)
%       S.eq_start: boolean to indicate if the agents start with the same
%       initial iterate (default = 0)
%       S.init: structure with details about the initial conditions
%         init.x: string to specify the initial condition to
%                 consider for the local iterates (x) (default = 'default')
%         init.D: real constant to use in the initial condition (cond_x <= D^2) (default = 1)
%         init.grad: string to choose the initial condition to
%                    consider for the local gradients (default = 'default')
%         init.E: real constant to use in the initial condition (cond_g <= E^2) (default = 1)
%         init.gamma: real coefficient to use in combined conditions (cond_x + gamma*cond_g <= D^2)
%                     (default = 1)
%       S.perf: string to specify the performance criterion to consider in PEP 
%               (default = 'default')
%       S.fctClass: string to specify the class of functions 
%                   (default = 'SmoothStronglyConvex')
%       S.fctParam: structure with the parameter values of the function class
%
% OUTPUT: 
%   Settings: structure with all the settings (specified and default values)

if ~isstruct(S)
    warning("settings should be provided in a single structure - default settings used");
    S = struct();
end

% Number of agents
if isfield(S,'n')
    set.n = S.n;
else % default
    set.n = 2;
end

% list of number of agents in different equivalence classes 
% (for symmetrized formulations)
if isfield(S,'nlist')
    set.nlist = S.nlist;
else % default
    set.nlist = 2;
end

% boolean: is n tending to infinity ?
if isfield(S,'ninf')
    set.ninf = S.ninf;
else % default
    set.ninf = 0;
end

% Number of iterations
if isfield(S,'t')
    set.t = S.t;
else % default
    set.t = 1;
end

% step-size(s)
if isfield(S,'alpha')
    if length(S.alpha) == 1 % use the same step-size for each iteration
        set.alpha = S.alpha*ones(1,set.t);
    elseif length(S.alpha) == set.t
        set.alpha = S.alpha;
    else
        error("the field 'alpha' should be of length 1 (constant step-size) or length t (time-varying step-size)");
    end
else % default
    set.alpha = 1*ones(1,set.t);
end

% ATC scheme for the algorithm ?
if isfield(S,'ATC')
    set.ATC = S.ATC;
else
    set.ATC = 0;
end

% averaging matrix description
if isfield(S,'avg_mat')
    % (a) Exact formulation (fixed network W)
    if all(size(S.avg_mat) == [set.n,set.n])
        set.type = 'exact';
        set.avg_mat = S.avg_mat;
    
    % (b) Spectral formulation for symmetric (generalized) doubly stochastic averaging matrices
    elseif length(S.avg_mat)==1 % SLEM description
        set.type = 'spectral_relaxed'; 
        set.avg_mat = [-S.avg_mat,S.avg_mat];
    elseif length(S.avg_mat)==2 % eigenvalue range
        set.type = 'spectral_relaxed'; 
        set.avg_mat = S.avg_mat;    
    else
        error("the field 'avg_mat' should be a matrix of size n x n (fixed averaging matrix), or a vector of length 1 (SLEM), or length 2 (range of eigenvalues)");
    end
elseif isfield(S,'lam') % alternative notation
    set.type = 'spectral_relaxed';
    if length(S.avg_mat)==1 % SLEM description
        set.avg_mat = [-S.lam,S.lam];
    elseif length(S.avg_mat)==2 % eigenvalue range
        set.avg_mat = S.lam;
    end
else % default
    set.type = 'spectral_relaxed';
    set.avg_mat = [-0.5,0.5]; % SLEM
end

% is the averaging matrix time-varying ?
if isfield(S,'tv_mat')
    set.tv_mat = S.tv_mat;
else % default
    set.tv_mat = 0;
end

% equal start for all the agents
if isfield(S,'eq_start')
    set.eq_start = S.eq_start;
else % default
    set.eq_start = 0;
end

% class of local functions
if isfield(S,'fctClass')
    set.fctClass = convertStringsToChars(S.fctClass);
else % default
    set.fctClass = 'SmoothStronglyConvex';
end

% parameters for the class of functions
if isfield(S,'fctParam')
    set.fctParam = S.fctParam;
else % default
    if strcmp(set.fctClass, 'SmoothStronglyConvex')
        set.fctParam.L = 1;
        set.fctParam.mu = 0.1;
    elseif strcmp(set.fctClass,'ConvexBoundedGradient')
        set.fctParam.R = 1;
    else
        set.fctParam.L=Inf;
        set.fctParam.mu=0;
        set.fctParam.D=Inf;
        set.fctParam.R=Inf;
    end
end

% initial conditions
if isfield(S,'init')
    set.init = lower(S.init);
    if ~isfield(S.init,'D')
        set.init.D = 1;
    end
    if ~isfield(S.init,'E')
        set.init.E = 1;
    end
    if ~isfield(S.init,'gamma')
        set.init.gamma = 1;
    end
    if ~isfield(S.init,'x')
        set.init.x = 'default';
    end
    if ~isfield(S.init,'grad')
        set.init.grad = 'default';
    end
else % default
    set.init.x = 'default';
    set.init.grad = 'default';
    set.init.D = 1;
    set.init.E = 1;
end

% Performance Criterion
if isfield(S,'perf')
    set.perf = lower(S.perf);
else % default
    set.perf = 'default';
end

% Summary of the Settings
% Settings.n = n; Settings.ninf = ninf; Settings.nlist = nlist; Settings.t = t; 
% Settings.alpha = alpha; Settings.ATC = ATC;
% Settings.type = type; Settings.avg_mat = avg_mat; Settings.tv_mat=tv_mat;
% Settings.eq_start = eq_start; Settings.init= init; Settings.perf=perf;
% Settings.fctClass = fctClass; Settings.fctParam = fctParam;

end