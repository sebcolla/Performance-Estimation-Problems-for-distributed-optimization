% Script to compute and plot the p-th percentile worst-case performance of EXTRA
% when the number of agents goes to infinity (n->inf).
% This performance is plotted as a function of the given percentile p
% (for different numbers of iterations t).
% See details in:
%    S. Colla and J. M. Hendrickx, "Exploiting Agent Symmetries for Performance Analysis of Distributed
%    Optimization Methods", 2024.

clear;

%% constant parameters
S.t = 1;
S.n = 2;
S.lam = 0.9;

S.eq_start = 0;
S.time_var_mat = 0;

S.init.x = 'navg_it_err_combined_s';

S.fctClass = 'SmoothStronglyConvex';
S.fctParam.L = 1;
S.fctParam.mu = 0.1;

S.perf = "navg_it_err_combined_s";

alphaspace = logspace(-4,-2,100);

out_alpha = cell(length(alphaspace),1);
wc_alpha = zeros(length(alphaspace),1);
filename = "data/DIGing_convrate_wc_alphaevol";

%% run PEP for EXTRA
il = 1;
for alpha=alphaspace
    fprintf("alpha = %1.2f \n",alpha);
    % spectral agent-dependent PEP
    S.alpha = alpha;
    S.init.gamma = alpha;
    out_alpha{il} = DIGing_agents(S);
    wc_alpha(il) = out_alpha{il}.WCperformance;
    save(filename)
end

%% PLOT results
%load('data/DGD_wc_lamevol.mat')
f1 = figure();
loglog(alphaspace,wc_alpha,'o-b','MarkerSize',2,'MarkerFaceColor', 'b','LineWidth',2); hold on;
ylim([0.999,1.001]);
xlabel("$\lambda(W)$ (SLEM)","FontSize",15,"Interpreter","Latex");
ylabel("$f(x_{\mathrm{av}}) - f(x^*)$","FontSize",14, 'Interpreter','Latex');
legend('Spectral PEP bound','Exact PEP bound for $\hat{W}^* = (1+\lambda) \frac{\mathbf{11}^T}{n} - \lambda I$','FontSize',13,'Location','NorthWest',"Interpreter","Latex");

%% save plot
% SAVE PDF
% set(f1,'PaperSize',[14, 10.1]); %set the paper size to what you want
% file_name = 'EXTRA_wb_it_ninf';
% print(f1,file_name,'-dpdf'); % then print it
