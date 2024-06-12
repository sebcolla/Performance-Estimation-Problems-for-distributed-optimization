% loading files
%load("new data/EXTRA_wc_lam09_all1");

% constant parameters
K = 10;
lam = 0.5;

eq_start = 0;
time_var_mat = 0;

init.type = 'uniform_bounded_iterr_local_grad*';


fctParam.L = 1;
fctParam.mu = 0.1;
filename = "new_data/EXTRA_wc_lam095_all2";

%% varying parameters
perf_favg = "fct_err_last_Navg";
perf_xavg = "Navg_last_it_err";

perf_fworst = "fct_err_last_worst";
perf_xworst = "it_err_last_worst";
perf_x80 = "it_err_last_percentile_worst";
perc = 0.8;

Nspace = [2,5,20:20:100];

%% AVERAGE performance
    save(filename);
    fun_xavg = @(alpha)fun_extra(2,K,alpha,lam,eq_start,time_var_mat,init,perf_xavg,fctParam);
    [alpha_opt_xavg,wc_EXTRA_xavg_sym,exitflag_xavg,output_xavg] = fminsearch(fun_xavg,0.5, optimset('TolX',5e-3,'MaxFunEvals',25));
   
    wc_EXTRA_xavg_agt = EXTRA_agents(2,K,alpha_opt_xavg,lam,time_var_mat,eq_start,init,perf_xavg,fctParam);
    fprintf("optimal alpha for xavg = %1.3f\n",alpha_opt_xavg);

    fun_favg = @(alpha)fun_extra(2,K,alpha,lam,eq_start,time_var_mat,init,perf_favg,fctParam);
    [alpha_opt_favg,wc_EXTRA_favg_sym,exitflag_favg,output_favg] = fminsearch(fun_favg,0.5, optimset('TolX',5e-3,'MaxFunEvals',25));
    wc_EXTRA_favg_agt = EXTRA_agents(2,K,alpha_opt_favg,lam,time_var_mat,eq_start,init,perf_favg,fctParam);
    
    wc_EXTRA_xw_alp_avg_agt2 = EXTRA_agents(2,K,alpha_opt_xavg,lam,time_var_mat,eq_start,init,perf_xworst,fctParam);
    wc_EXTRA_fw_alp_avg_agt2 = EXTRA_agents(2,K,alpha_opt_favg,lam,time_var_mat,eq_start,init,perf_fworst,fctParam);

    save(filename);
%%
ni = 1;
for n=Nspace
    fprintf("N = %d\n",n);
    Nperc = [(1-perc)*n,1,perc*n-1];
    Nworst = [1,n-1];

    % WORST with alpha_avg*
    % symmetrized
     wc_EXTRA_xw_alp_avg{ni} = EXTRA_symmetrized(Nworst,K,alpha_opt_xavg,lam,time_var_mat,eq_start,init,perf_xworst,fctParam);
     wc_EXTRA_fw_alp_avg{ni} = EXTRA_symmetrized(Nworst,K,alpha_opt_favg,lam,time_var_mat,eq_start,init,perf_fworst,fctParam);
     
    % 80 perc with alpha_avg*
     wc_EXTRA_80x_alp_avg{ni} = EXTRA_symmetrized(Nperc,K,alpha_opt_xavg,lam,time_var_mat,eq_start,init,perf_x80,fctParam);
    save(filename);
    
    % WORST with optimized alpha_w*(n)
    fun_xw{ni} = @(alpha)fun_extra(Nworst,K,alpha,lam,eq_start,time_var_mat,init,perf_xworst,fctParam);
    [alpha_opt_xw(ni),wc_EXTRA_xw(ni),exitflag_xw{ni},output_xw{ni}] = fminsearch(fun_xw{ni},alpha_opt_xavg*sqrt(2)/sqrt(n), optimset('TolX',5e-3,'MaxFunEvals',25));
    
    fun_fw{ni} = @(alpha)fun_extra(Nworst, K,alpha,lam,eq_start,time_var_mat,init,perf_fworst,fctParam);
    [alpha_opt_fw(ni),wc_EXTRA_fw(ni),exitflag_fw{ni},output_fw{ni}] = fminsearch(fun_fw{ni},alpha_opt_favg*sqrt(2)/sqrt(n), optimset('TolX',5e-3,'MaxFunEvals',25));

    save(filename);
    
    % avg with alpha_w*(n)
    wc_EXTRA_xavg_alphw{ni} = EXTRA_agents(2,K,alpha_opt_xw(ni),lam,time_var_mat,eq_start,init,perf_xavg,fctParam);
    wc_EXTRA_favg_alphw{ni} = EXTRA_agents(2,K,alpha_opt_fw(ni),lam,time_var_mat,eq_start,init,perf_favg,fctParam);
    save(filename);
    
    % 80 perc with optimized alpha_80*(n)
    if n > 2
        fun_x80{ni} = @(alpha)fun_extra(Nperc,K,alpha,lam,eq_start,time_var_mat,init,perf_x80,fctParam);
        [alpha_opt_x80(ni),wc_EXTRA_x80(ni),exitflag_x80{ni},output_x80{ni}] = fminsearch(fun_x80{ni},alpha_opt_xavg*sqrt(2)/sqrt(n), optimset('TolX',5e-3,'MaxFunEvals',25));
    end
    save(filename);
    ni = ni+1;
end

function wc = fun_extra(N,K,alpha,lam,eq_start,time_var_mat,init,perf,fctParam) 
     res = EXTRA_symmetrized(N,K,alpha,lam,eq_start,time_var_mat,init,perf,fctParam);
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