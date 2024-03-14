% constant parameters
lam = 0.5;
Ninf = 1;

eq_start = 0;
time_var_mat = 0;

init.type = 'uniform_bounded_iterr_local_grad*';


fctParam.L = 1;
fctParam.mu = 0.1;

perf = "it_err_last_percentile_worst";

alpha = 0.78;

Klist = [15];
perc = [0.01,0.1:0.1:0.7,0.75,0.8,0.85,0.875,0.9,0.925,0.95,0.96,0.97,0.98,0.99];

d1 = cell(length(Klist),length(perc));
wc = zeros(length(Klist),length(perc));
filename = "new_data/EXTRA_wc_lam05_Ninf_00";

%% run PEP for EXTRA
ik = 1; 
for K=Klist
    ip = 1;
    for p1=perc
        fprintf("K = %d, p=%.2f \n",K,p1);
        Nproplist = [1-p1,0,p1];
        d1{ik,ip} = EXTRA_symmetrized(Nproplist,K,alpha,lam,time_var_mat,eq_start,init,perf,fctParam,Ninf);
        wc(ik,ip) = d1{ik,ip}.WCperformance;
        ip = ip+1;
        save(filename)
    end
    ik = ik+1;
end

%% PLOT results
%load('C:\Users\secolla\Documents\UCL\PhD\code\CLEAN codes\symmetrized form\new data\EXTRA_wc_lam05_Ninf4.mat')
f1 = figure();
plot([perc,1]*100,[wc(1:4,:),2*ones(4,1)],'.-','LineWidth',2,'MarkerSize',15); hold on;
xlabel("k (percentage of agents)","FontSize",14,"Interpreter","Latex");
ylabel("\textbf{k--th} ~Percentile Performance","FontSize",14,"Interpreter","Latex");
ylim([0,1])
legend("$t=10$","$t=15$","$t=20$","$t=25$","FontSize",12,"Interpreter","Latex","Location","NorthWest");

%% save plot
% SAVE PDF
% set(f1,'PaperSize',[14, 10.1]); %set the paper size to what you want
% file_name = 'EXTRA_wb_it_Ninf'; %wc_lamevol_N3_K10_eqref_update';
% print(f1,sprintf('../../hybrid sym form/plots_pdf/%s',file_name),'-dpdf'); % then print it
