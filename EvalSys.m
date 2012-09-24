function [eer,mindcf_old,mindcf_new] = EvalSys(FinalScores,answers)
%% Demo code to compute min DCF using the SRE2008 and SRE2010 settings.

% load the score files
% true_scores = textread('./true','%f');
% imp_scores = textread('./imp','%f');

true_scores = FinalScores(answers == 1);
imp_scores = FinalScores(answers == 0);

% setup for SRE08 and SRE10 style minDCF calculation.
Cmiss_SRE08 = 10;
Cfa_SRE08 = 1;
Ptarget_SRE08 = 0.01;

Cmiss_SRE10 = 1;
Cfa_SRE10 = 1;
Ptarget_SRE10 = 0.001;

dcfWeights_SRE08=[Cmiss_SRE08*Ptarget_SRE08 Cfa_SRE08*(1-Ptarget_SRE08)];
dcfWeights_SRE10=[Cmiss_SRE10*Ptarget_SRE10 Cfa_SRE10*(1-Ptarget_SRE10)];

[~,~,eer,mindcf_new] = rocchdet(true_scores,imp_scores,dcfWeights_SRE10);
[~,~,~,mindcf_old] = rocchdet(true_scores,imp_scores,dcfWeights_SRE08);

mindcf_new = mindcf_new/min(dcfWeights_SRE10);
mindcf_old = mindcf_old/min(dcfWeights_SRE08);

eer = eer*100;

disp(['EER : ',sprintf('%0.5f',eer)]);
disp(['minDCF_old : ',sprintf('%0.5f',mindcf_old)]);
disp(['minDCF_new : ',sprintf('%0.5f',mindcf_new)]);

