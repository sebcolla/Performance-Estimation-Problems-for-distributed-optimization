% Script to compute the worst-case performance of EXTRA in different settings
% and analyze their evolution with the number of agents n:
%   - agent-average last iterates error,
%   - functional error at the agent-average last iterate,
%   - worst agent last iterate error,
%   - functional error at the worst agent last iterate,
%   - 80-th percentile of the agents last iterate error,
% See details in:
%    S. Colla and J. M. Hendrickx, "Exploiting Agent Symmetries for Performance Analysis of Distributed
%    Optimization Methods", 2024.

clear all;

%% constant parameters
S.t = 15;
S.lam = 0.5;

S.eq_start = 0;
S.tv_mat = 0;

S.init.x = 'uniform_bounded_it_err';
S.init.grad = 'uniform_bounded_grad*';

S.fctClass = 'SmoothStronglyConvex';
S.fctParam.L = 1;
S.fctParam.mu = 0.1;

filename = "data/EXTRA_wc_t15_lam05_test";

%% AVERAGE performance
%save(filename);
% compute optimal step-size for average performance metrix
S.n = 2;

S.perf = "navg_last_it_err";
fun_xavg = @(alpha)fun_extra(S,alpha);
[alpha_opt_xavg,wc_EXTRA_xavg_sym,exitflag_xavg,output_xavg] = fminsearch(fun_xavg,0.5, optimset('TolX',5e-3,'MaxFunEvals',25));
fprintf("optimal alpha for xavg = %1.3f\n",alpha_opt_xavg);

S.alpha = alpha_opt_xavg;
wc_EXTRA_xavg_agt = EXTRA_agents(S);

S.perf = 'it_err_last_worst';
wc_EXTRA_xw_alph_avg_agt2 = EXTRA_agents(S);

S.perf = "fct_err_last_navg";
fun_favg = @(alpha)fun_extra(S,alpha);
[alpha_opt_favg,wc_EXTRA_favg_sym,exitflag_favg,output_favg] = fminsearch(fun_favg,0.5, optimset('TolX',5e-3,'MaxFunEvals',25));
S.alpha = alpha_opt_favg;
wc_EXTRA_favg_agt = EXTRA_agents(S);
S.perf = 'fct_err_last_worst';
wc_EXTRA_fw_alph_avg_agt2 = EXTRA_agents(S); 

save(filename);

%% Worst agent and percentile performance (for different system sizes n)
nspace = [2,5,20:20:100];
perc = 0.8;
ni = 1; 
for n=nspace
    fprintf("n = %d\n",n);
    
    % WORST agent performance with alpha_avg* (it_err)
    S.nlist = [1,n-1];
    S.alpha = alpha_opt_xavg; S.perf = 'it_err_last_worst';
    wc_EXTRA_xw_alph_avg{ni} = EXTRA_symmetrized(S);
    
    % WORST agent performance with optimized alpha_w*(n) (it_err)
    fun_xw{ni} = @(alpha)fun_extra(S,alpha);
    [alpha_opt_xw(ni),wc_EXTRA_xw(ni),exitflag_xw{ni},output_xw{ni}] = fminsearch(fun_xw{ni},alpha_opt_xavg*sqrt(2)/sqrt(n), optimset('TolX',5e-3,'MaxFunEvals',25));
    % avg perf with alpha_w*(n)
    S.alpha = alpha_opt_xw(ni); S.perf = "navg_last_it_err";
    wc_EXTRA_xavg_alphw{ni} = EXTRA_agents(S);

    % WORST agent performance with alpha_avg* (fct_err)
    S.alpha = alpha_opt_favg; S.perf = 'fct_err_last_worst';
    wc_EXTRA_fw_alph_avg{ni} = EXTRA_symmetrized(S);
    
    % WORST-agent performance with optimized alpha_w*(n) (fct_err)
    fun_fw{ni} = @(alpha)fun_extra(S,alpha);
    [alpha_opt_fw(ni),wc_EXTRA_fw(ni),exitflag_fw{ni},output_fw{ni}] = fminsearch(fun_fw{ni},alpha_opt_favg*sqrt(2)/sqrt(n), optimset('TolX',5e-3,'MaxFunEvals',25));
    % avg perf with alpha_w*(n)
    S.alpha = alpha_opt_fw(ni); S.perf = "fct_err_last_navg";
    wc_EXTRA_xavg_alphw{ni} = EXTRA_agents(S);

    % 80-percentile performance with alpha_avg*
    S.nlist = [(1-perc)*n,1,perc*n-1];
    S.alpha = alpha_opt_xavg; S.perf = 'it_err_last_percentile_worst';
    wc_EXTRA_80x_alph_avg{ni} = EXTRA_symmetrized(S);
    
    save(filename);
    
    % 80 percentile performance with optimized alpha_80*(n)
    if n > 2
        fun_x80{ni} = @(alpha)fun_extra(S,alpha);
        [alpha_opt_x80(ni),wc_EXTRA_x80(ni),exitflag_x80{ni},output_x80{ni}] = fminsearch(fun_x80{ni},alpha_opt_xavg*sqrt(2)/sqrt(n), optimset('TolX',5e-3,'MaxFunEvals',25));
    end
    save(filename);
    ni = ni+1;
end

function wc = fun_extra(Settings,alpha) %N,K,alpha,lam,eq_start,time_var_mat,init,perf,fctParam)
Settings.alpha = alpha;
res = EXTRA_symmetrized(Settings);
if alpha < 0
    wc = Inf;
    return
end
if res.solverDetails.info == "Successfully solved (MOSEK)"
    wc = res.WCperformance;
else
    wc = Inf;
end
fprintf("alpha = %1.3f, gives perf = %1.4f \n",alpha,wc);
end