% Script to compute and plot the p-th percentile worst-case performance of EXTRA
% when the number of agents goes to infinity (n->inf).
% This performance is plotted as a function of the given percentile p
% (for different numbers of iterations t).
% See details in:
%    S. Colla and J. M. Hendrickx, "Exploiting Agent Symmetries for Performance Analysis of Distributed
%    Optimization Methods", 2024.

clear;

%% constant parameters
S.t = 10;
S.n = 2;

S.eq_start = 1;
S.time_var_mat = 0;

S.init.x = 'bounded_navg_it_err';

S.fctClass = 'ConvexBoundedGradient';
S.fctParam.R = 1;

S.perf = "fct_err_tavg_navg";

S.alpha = 1/sqrt(S.t);

lamspace = [0:0.05:0.99];

out_lam = cell(length(lamspace),1);
wc_lam = zeros(length(lamspace),1);
out_exact = cell(length(lamspace),1);
wc_exact = zeros(length(lamspace),1);
filename = "data/DGD_wc_lamevol";

%% solve PEP for DGD for different lam values
il= 1;
for lam=lamspace
    fprintf("lam = %1.2f \n",lam);
    % spectral agent-dependent PEP
    S.avg_mat = lam;
    out_lam{il} = DGD_agents(S);
    wc_lam(il) = out_lam{il}.WCperformance;
    fprintf("The worst-case averaging matrix is ")
    out_lam{il}.Wh.W
    
    % exact agent-dependent PEP with the worst-case
    % averaging matrix from the spectral PEP.
    S.avg_mat = out_lam{il}.Wh.W;
    out_exact{il} = DGD_agents(S);
    wc_exact(il) = out_exact{il}.WCperformance;
    il = il+1;
    save(filename)
end

%% PLOT results
%load('data/DGD_wc_lamevol.mat')
f1 = figure();
plot(lamspace,wc_lam,'^-b','MarkerSize',4,'MarkerFaceColor', 'b','LineWidth',2.5); hold on;
plot(lamspace,wc_exact,'s-g','MarkerSize',5.75,'MarkerFaceColor', 'g','LineWidth',1.5);
ylim([0.3,1.1]);
xlabel("$\lambda(W)$ (SLEM)","FontSize",15,"Interpreter","Latex");
ylabel("$f(x_{\mathrm{av}}) - f(x^*)$","FontSize",14, 'Interpreter','Latex');
legend('Spectral PEP bound','Exact PEP bound for $\hat{W}^* = (1+\lambda) \frac{\mathbf{11}^T}{n} - \lambda I$','FontSize',13,'Location','NorthWest',"Interpreter","Latex");

%% save plot
% SAVE PDF
% set(f1,'PaperSize',[14, 10.1]); %set the paper size to what you want
% file_name = 'EXTRA_wb_it_ninf';
% print(f1,file_name,'-dpdf'); % then print it
