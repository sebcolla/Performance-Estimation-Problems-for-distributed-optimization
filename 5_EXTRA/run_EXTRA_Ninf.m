% Script to compute and plot the p-th percentile worst-case performance of EXTRA
% when the number of agents goes to infinity (n->inf).
% This performance is plotted as a function of the given percentile p 
% (for different numbers of iterations t).
% See details in:
%    S. Colla and J. M. Hendrickx, "Exploiting Agent Symmetries for Performance Analysis of Distributed
%    Optimization Methods", 2024.

clear all;

%% constant parameters
S.lam = 0.5;
S.ninf = 1;

S.eq_start = 0;
S.time_var_mat = 0;

S.init.x = 'uniform_bounded_it_err';
S.init.grad = 'uniform_bounded_grad*';

S.fctClass = 'SmoothStronglyConvex';
S.fctParam.L = 1;
S.fctParam.mu = 0.1;

S.perf = "it_err_last_percentile_worst";

S.alpha = 0.78;

tlist = [10,15];
perc = [0.01,0.1:0.1:0.7,0.75,0.8,0.85,0.875,0.9,0.925,0.95,0.96,0.97,0.98,0.99];

d1 = cell(length(tlist),length(perc));
wc = zeros(length(tlist),length(perc));
filename = "data/EXTRA_wc_lam05_ninf";

%% run PEP for EXTRA
ik = 1; 
for t=tlist
    ip = 1;
    S.t = t;
    for p1=perc
        fprintf("t = %d, p=%.2f \n",t,p1);
        nproplist = [1-p1,0,p1];
        S.nlist = nproplist;
        d1{ik,ip} = EXTRA_symmetrized(S);
        wc(ik,ip) = d1{ik,ip}.WCperformance;
        ip = ip+1;
        save(filename)
    end
    ik = ik+1;
end

%% PLOT results
%load('data\EXTRA_wc_lam05_ninf.mat')
f1 = figure();
plot([perc,1]*5,[wc,100*ones(length(tlist),1)],'.-','LineWidth',2,'MarkerSize',15); hold on;
xlabel("k (percentage of agents)","FontSize",14,"Interpreter","Latex");
ylabel("\textbf{k--th} ~Percentile Performance","FontSize",14,"Interpreter","Latex");
ylim([0,1])
legend("$t=10$","$t=15$","$t=20$","$t=25$","FontSize",12,"Interpreter","Latex","Location","NorthWest");

%% save plot
% SAVE PDF
% set(f1,'PaperSize',[14, 10.1]); %set the paper size to what you want
% file_name = 'EXTRA_wb_it_Ninf'; %wc_lamevol_N3_K10_eqref_update';
% print(f1,sprintf('../../hybrid sym form/plots_pdf/%s',file_name),'-dpdf'); % then print it
