function [n,t,alpha,type,mat,tv_mat,eq_start,init,perf,fctClass,fctParam] = extractSettings(S)
% The 'extractSettings' function extracts the settings provided in S, and
% assign default values for those not specified
if ~isstruct(S)
    warning("settings should be provided in a single structure - default settings used");
    S = struct();
end
disp('yooooo')
if isfield(S,'n')
    n = S.n;
else % default
    n = 2;
end

if isfield(S,'t')
    t = S.t;
else % default
    t = 1;
end

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

if isfield(S,'avg_mat')
    % (a) Exact formulation (fixed network W)
    if all(size(S.avg_mat) == [n,n])
        type = 'exact';
        mat = S.avg_mat;
    
    % (b) Spectral formulation for symmetric (generalized) doubly stochastic averaging matrices
    elseif length(S.avg_mat)==1 % SLEM description
        type = 'spectral_relaxed'; 
        mat = [-S.avg_mat,S.avg_mat];
    elseif length(S.avg_mat)==2 % eigenvalue range
        type = 'spectral_relaxed'; 
        mat = S.avg_mat;    
    else
        error("the field 'avg_mat' should be a matrix of size n x n (fixed averaging matrix), or a vector of length 1 (SLEM), or length 2 (range of eigenvalues)");
    end
else % default
    type = 'spectral_relaxed';
    mat = 0.5; % SLEM
end

if isfield(S,'tv_mat')
    tv_mat = S.tv_mat;
else % default
    tv_mat = 0;
end

if isfield(S,'eq_start')
    eq_start = S.eq_start;
else % default
    eq_start = 0;
end

if isfield(S,'fctClass')
    fctClass = convertStringsToChars(S.fctClass);
else % default
    fctClass = 'SmoothStronglyConvex';
end

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

if isfield(S,'init')
    init = lower(S.init);
else % default
    init.type = "bounded_navg_it_err"; % to adapt?
    init.D = 1;
end

if isfield(S,'perf')
    perf = lower(S.perf);
else % default
    perf = "fct_err_last_navg"; % to adapt ?
end

end