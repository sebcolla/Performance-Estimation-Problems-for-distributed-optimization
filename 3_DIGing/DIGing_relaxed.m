function out = DIGing_relaxed(Settings)
% Compute the worst-case performance of t steps of DIGing [1] under the 'Settings' provided and
% using the relaxed agent-independent PEP formulation from [2].
% REQUIREMENTS: YALMIP toolbox with Mosek solver.
% INPUT:
%   Settings: structure with all the settings to use in the PEP for DIGing. 
%   The structure can include the following fields:
%   (unspecified fields will be set to a default value)
%       Settings.t: number of iterations (default = 1)
%       Settings.alpha: step-size (scalar or vector of t elements) (default = 1)
%       Settings.avg_mat: averaging matrix spectral description (one of the following options)
%           - the second-largest eigenvalue modulus (scalar)
%           - range of eigenvalues for the averaging matrix (except one): [lb, ub] with -1 < lb <= ub < 1.
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
%                           the local gradients (default = 'none')
%                init.E:    real constant to use in the initial condition (cond_g <= E^2) (default = 1)
%                init.gamma: real coefficient to use in combined conditions (cond_x + gamma*cond_g <= D^2)
%                           (default = 1)
%       Settings.perf: string to specify the performance criterion to consider in PEP
%                      (default = 'fct_err_last_navg')
%       Settings.fctClass: string to specify the class of functions (default = 'SmoothStronglyConvex')
%       Settings.fctParam: structure with the parameter values of the function class
%
% OUTPUT: structure with details about the worst-case solution of the PEP, including
%   WCperformance : worst-case value.
%   solverDetails : details given by the Mosek solver
%   Settings:       structure with all the settings used in the PEP
%                   (including all the default values that have been set)
%
% References:
%   [1] A. Nedic, A. Olshevsky, and W. Shi, “Achieving geometric convergence
%       for distributed optimization over time-varying graphs,” SIAM Journal on
%       Optimization, 2016.
%   [2] Sebastien Colla and Julien M. Hendrickx. Automated performance estimation 
%       for decentralized optimization via network size independent problems. IEEE, CDC 2022.

verbose = 1;            % Print the problem set up and the results
verbose_solv = 0;       % Print details of the solver
trace_heuristic = 0;    % heuristic to minimize the dimension of the worst-case

%%% Set up performance estimation settings %%%
if nargin == 1
    Settings = extractSettings(Settings);
else
    warning("settings should be provided in a single structure - default settings used")
    Settings = extractSettings(struct());
end
t=Settings.t; alpha=Settings.alpha;
type=Settings.type; lamW=Settings.avg_mat; tv_mat=Settings.tv_mat;
eq_start=Settings.eq_start; init=Settings.init; perf=Settings.perf;
fctClass=Settings.fctClass; fctParam=Settings.fctParam;

if ~strcmp(fctClass,'SmoothStronglyConvex') && ~strcmp(fctClass,'ConvexBoundedGradient')
    warning("Class of functions not supported. PEP solution computed for fctClass = 'SmoothStronglyConvex' (with L=1, mu=0.1)");
    % other classes of function requires different interpolation conditions
end
assert(strcmp(type,'spectral_relaxed'),"DGD_symmetrized only applies to spectral description of the averaging matrix (range of eigenvalues)");

if verbose
    fprintf("Settings provided for the PEP:\n");
    fprintf("t=%d, alpha=%1.2f, type=%s, tv_mat=%d, eq_start=%d,\ninit_x=%s, init_grad=%s, perf=%s, fctClass=%s,\n",t,alpha(1),type,tv_mat,eq_start,init.x,init.grad,perf,fctClass);
    fprintf('avg_mat = ['); fprintf('%g ', lamW); fprintf(']\n');
    fprintf("------------------------------------------------------------------------------------------\n");
end

%% Defining the coefficient vectors to access SDP variables easily
% Gb = Pb^T Pb and Gp = Pp^T Pp
% Pb = [xb0 gb0...gb^t gcb]
% Pp = [xp0 gp0...gp^t Wxp0...Wxp^t-1 sp0 Wsp0...Wsp^t-1 gcp gsp]
% F = [f0..ft fc]

dimGb = t+3;
dimGp = 3*(t+1)+1;
dimF = t+2;
nbPts = t+3;

if verbose
    fprintf("Problem Size = (%dx%d), (%dx%d) \n", dimGb,dimGb,dimGp,dimGp);
end

% Optimum coeff
xsb   = zeros(dimGb,1);
%xsb(end-1) = 1; % no need because we suppose xb* = 0
xsp   = zeros(dimGp,1); % optimal solution achieves consensus
gsb   = zeros(dimGb,1); % no gradient in the consensus direction
gsp   = zeros(dimGp,1);
gsp(end) = 1;
fs   = zeros(dimF,1);
%fs(end) = 1; % no need because we suppose f(xb*) = 0

% Initial iterates coeff
xb  = zeros(dimGb, t+1);  xb(1,1) = 1;
xp  = zeros(dimGp, t+1); 
if ~eq_start
    xp(1,1) = 1;
end
gb  = zeros(dimGb, t+1);  gb(2:2+t,:)    = eye(t+1);
gp  = zeros(dimGp, t+1);  gp(2:2+t,:) = eye(t+1);
sb  = zeros(dimGb, t+1);  sb(2,1) = 1; 
if (strcmp(perf,'navg_it_err_combined_s') || strcmp(perf,'2')) && (strcmp(init.x,'navg_it_err_combined_s') || strcmp(init.x,'2')) 
    % to use for CONV RATE analysis of DIGing (with gamma = alpha)
    sp  = zeros(dimGp, t+1);  sp(2*t+3,1) = 1;  % sum_i si0 = sum_i gi0
else % s0 = g0
    sp  = zeros(dimGp, t+1);  sp(2,1) = 1; 
end
f  = zeros(dimF,t+1);
f(1:t+1,:) = eye(t+1);

% Consensus iterates coeff
Wxp = zeros(dimGp, t); Wxp(t+3:2*t+2,:)        = eye(t);
Wsp = zeros(dimGp, t); Wsp(2*t+4:3*t+3,:)  = eye(t);

% DIGing iterates coeff
for k = 1:t
    xb(:,k+1) = xb(:,k) - alpha(k)*sb(:,k);
    xp(:,k+1) = Wxp(:,k) - alpha(k)*sp(:,k);
    sb(:,k+1) = sb(:,k) + gb(:,k+1) - gb(:,k);
    sp(:,k+1) = Wsp(:,k) + gp(:,k+1) - gp(:,k);
end

% coeff for the point common to all agents, used in the performance criterion
fc   = zeros(dimF,1); fc(t+2)  = 1;
gcb  = zeros(dimGb,1); gcb(t+3) = 1;
gcp  = zeros(dimGp,1); gcp(2*t+3) = 1;

if strcmp(perf,'fct_err_tavg_navg') || strcmp(perf,'8')
    xcb = mean(xb,2);  % average over all iterations and all agents
else
    xcb = xb(:,t+1);   % average over all agents of the last iterate
end
xcp = zeros(dimGp,1);  % same point for all agents (xcp = 0)

% set of points to interpolate for the local functions
Xb_interp = [xsb, xb, xcb];
Xp_interp = [xsp, xp, xcp];
Gb_interp = [gsb, gb, gcb];
Gp_interp = [gsp, gp, gcp];
F_interp = [fs, f, fc];

% set of pair of points to interpolate for the averaging matrix
Xcons = [xp(:,1:t), sp(:,1:t)];
WXcons = [Wxp, Wsp];
kcons = length(Xcons(1,:));

%% PEP problem

% (1) VARIABLES
Gb = sdpvar(dimGb);
Gp = sdpvar(dimGp);
F = sdpvar(1,dimF);

% (2) SDP constraint G>= 0
cons = (Gb >= 0);
cons = cons + (Gp >= 0);

% (3) INTERPOLATION conditions for local functions fi
for i = 1:nbPts
    for j = 1:nbPts
        xbi = Xb_interp(:,i); xpi = Xp_interp(:,i);
        gbi = Gb_interp(:,i); gpi = Gp_interp(:,i);
        fi = F_interp(:,i);
        xbj = Xb_interp(:,j); xpj = Xp_interp(:,j);
        gbj = Gb_interp(:,j); gpj = Gp_interp(:,j);
        fj = F_interp(:,j);
        switch fctClass
            case 'ConvexBoundedGradient'
                if i ~= j
                    cons = cons + ( (F*(fi-fj) + gbi.'*Gb*(xbj-xbi) + gpi.'*Gp*(xpj-xpi) )<=0);
                else
                    if ~(all(gbi==0) && all(gpi==0))
                        cons = cons + (gbi.'*Gb*gbi + gpi.'*Gp*gpi <= fctParam.R^2);
                    end
                end
            otherwise  %'SmoothStronglyConvex'
                if i ~= j
                    L = fctParam.L; mu = fctParam.mu;
                    cons = cons + ( (F*(fj-fi) + gbj.'*Gb*(xbi-xbj) + gpj.'*Gp*(xpi-xpj) +...
                        1/(2*(1-mu/L)) *(1/L*((gbi-gbj).'*Gb*(gbi-gbj) + (gpi-gpj).'*Gp*(gpi-gpj)) + ...
                        mu *( (xbj - xbi).'*Gb*(xbj - xbi) + (xpj - xpi).'*Gp*(xpj - xpi) ) - ...
                        2*mu/L*( (xbj - xbi).'*Gb*(gbj - gbi) + (xpj - xpi).'*Gp*(gpj - gpi) ) ) ) <= 0 );
                    % f_i >= f_j + g_j^T (xi-xj) + 1/2/L ||g_i-g_j||^2 si m = 0
                end
        end
    end
end
      

% (4) INTERPOLATION conditions for the averaging matrix W, used in consensus steps
if ~tv_mat % time-constant averaging matrix
    SP1 = sdpvar(kcons);
    cons = cons + (SP1 >= 0);
    for k1 = 1:kcons
        for k2 = 1:kcons
            cons = cons + (SP1(k1,k2) == (-lamW(1)*lamW(2)*Xcons(:,k1).'*Gp*Xcons(:,k2) + (lamW(1)+lamW(2))*Xcons(:,k1).'*Gp*WXcons(:,k2) - WXcons(:,k1).'*Gp*WXcons(:,k2)));
        end
    end
    if t>1 % Symmetry condition
        cons = cons + (WXcons.'*(Gp)*Xcons - Xcons.'*(Gp)*WXcons == 0); %(X-Xb)^T(Wx-Xb) = (Wx-Xb)^T(X-Xb).
    end
else % time-varying averaging matrix
    for k = 1:kcons
         cons = cons + ( (-lamW(1)*lamW(2)*Xcons(:,k).'*Gp*Xcons(:,k) + (lamW(1)+lamW(2))*Xcons(:,k).'*Gp*WXcons(:,k) - WXcons(:,k).'*Gp*WXcons(:,k)) >= 0);
    end
end

% (5) INITIAL CONDITIONS
% Initial condition for x0
switch init.x
    case {'bounded_navg_it_err','0'}      % avg_i ||xi0 - xs||^2 <= D^2
        cons = cons + ((xb(:,1)-xsb).'*Gb*(xb(:,1)-xsb) + (xp(:,1)-xsp).'*Gp*(xp(:,1)-xsp) <= init.D^2);
    case {'navg_it_err_combined_s','2'}   % to use for CONV RATE analysis of DIGing (with gamma = alpha)
        % (avg_i ||xi0 - xs||^2) + gamma* (avg_i ||si0 - avg_i(gi0)||^2) <= D^2
        cons = cons + ((xb(:,1)-xsb).'*Gb*(xb(:,1)-xsb) + (xp(:,1)-xsp).'*Gp*(xp(:,1)-xsp) + init.gamma*( (sp(:,1)).'*Gp*(sp(:,1)) ) <= init.E^2 );
    otherwise % default (for DGD) is 'bounded_navg_it_err'
        cons = cons + ((xb(:,1)-xsb).'*Gb*(xb(:,1)-xsb) + (xp(:,1)-xsp).'*Gp*(xp(:,1)-xsp) <= init.D^2);
end

% Initial condition for g0
switch init.grad
    case {'bounded_navg_grad0','0'}       % avg_i ||gi0||^2 <= E^2
        cons = cons + (gb(:,1).'*(Gb)*gb(:,1) + gp(:,1).'*(Gp)*gp(:,1) <= init.E^2);
    case {'bounded_grad0_cons_err','2'}    % avg_i ||gi0 - avg_i(gi0)||^2 <= E^2
        % bound on the distance of the initial gradient to their average:
        cons = cons + ((gp(:,1)).'*Gp*(gp(:,1))  <= init.E^2 );
    case {'bounded_navg_grad*','3'}       % avg_i ||gi(x*)||^2 <= E^2
        cons = cons + (gsb(:,1).'*(Gb)*gsb(:,1) + gsp(:,1).'*(Gp)*gsp(:,1) <= init.E^2);
    otherwise % default is 'none'
end

% (6) OBJECTIVE FUNCTION of the PEP
switch perf
    case {'navg_last_it_err','0'}       % 1/n sum_i ||x_i^t - x*||2
        obj = (xb(:,t+1)-xsb).'*Gb*(xb(:,t+1)-xsb) + (xp(:,t+1)-xsp).'*Gp*(xp(:,t+1)-xsp);
    case {'navg_it_err_combined_s','2'} % (avg_i ||xi^k - xs||^2) + gamma* (avg_i ||si^k - avg_i(gi^k)||^2)
        % to use for CONV RATE analysis of DIGing (with gamma = alpha)
        obj = (xb(:,t+1)-xsb).'*Gb*(xb(:,t+1)-xsb) + (xp(:,t+1)-xsp).'*Gp*(xp(:,t+1)-xsp) + init.gamma*( (sp(:,t+1)).'*Gp*(sp(:,t+1)) );
    case { 'fct_err_last_navg', 'fct_err_tavg_navg','6','8'}
        obj = F*(fc - fs);
    otherwise % default is 'fct_err_last_navg'
        warning("Performance metric not supported. Default one used: 'fct_err_last_navg'");
        obj = F*(fc - fs);
end

% (7) Solve the SDP PEP
solver_opt      = sdpsettings('solver','mosek','verbose',verbose_solv);
solverDetails   = optimize(cons,-obj,solver_opt);

if verbose
    fprintf("Solver output %7.5e, \t Solution status %s \n",double(obj),solverDetails.info);
end

% Trace Heuristic
if trace_heuristic
    cons = cons + (obj >= wc-1e-5);
    solverDetails  = optimize(cons,trace(G),solver_opt);
    wc = double(obj);
    if verbose
        fprintf("Solver output after Trace Heurisitc: %7.5e, \t Solution status %s \n",wc, solverDetails.info);
    end
end

% OUTPUT
out.solverDetails = solverDetails;
out.WCperformance = double(obj);
out.Settings = Settings;

end
