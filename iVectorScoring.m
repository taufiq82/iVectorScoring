function iVectorScoring(varargin)
%% Example command:
% This should work:
% iVectorScoring('iVecDev','iVectorDEV','iVecTrain','iVectorTrain','iVecTest','iVectorTest','DevSpkrID','DEV_spkr_id','TrialsIndex','Trials_MALE_tel_tel_index','OutScores','OutputScores','iVecSize','400','Scoring','PLDA_250','DimReduc','LDA_250','Norm','RG')
% also with lists: (this wont work here)
%iVectorScoring('iVecDev','/home/txh085000/temp/iVecDevList','iVecTrain','/home/txh085000/temp/iVecTrainList','iVecTest','/home/txh085000/temp/iVecTestList','DevSpkrID','/home/txh085000/temp/IdxSpkrLDA_m8','TrialsIndex','/home/txh085000/temp/TrialsList_FEMALE_Dev_U2_K2_index','OutScores','/tmp/FinalScores_Baseline','iVecSize','400','Scoring','CD','Precision','single','ReadMode','list','ResultFile','/tmp/res_baseline')

p = inputParser; % Create an instance of the class.
p.addParamValue('iVecDev', '',@ischar);
p.addParamValue('iVecTrain', '',@ischar);
p.addParamValue('iVecTest', '',@ischar);
p.addParamValue('iVecWCCN', '',@ischar);

p.addParamValue('DevSpkrID', '',@ischar);
p.addParamValue('WCCNSpkrID', '',@ischar);
p.addParamValue('TrainSpkrID', '',@ischar);
p.addParamValue('UseTrainForDEV', '0',@ischar);

p.addParamValue('TrainList', '',@ischar);
p.addParamValue('TestList', '',@ischar);
p.addParamValue('Trials', '',@ischar);
p.addParamValue('TrialsIndex', '',@ischar);
p.addParamValue('Ans', '',@ischar);
p.addParamValue('OutScores', '',@ischar);
p.addParamValue('ResultFile', '',@ischar);
p.addParamValue('Precision', 'double',@ischar);

p.addParamValue('iVecSize', '',@ischar);
p.addParamValue('Scoring', '',@ischar); % PLDA or CD
p.addParamValue('DimReduc', '',@ischar); % LDA_200 etc.
p.addParamValue('Norm', '',@ischar); % WCCN/Lnorm/RG (Radial Gaussianization)
p.addParamValue('ReadMode', 'matrix',@ischar); % read matrix or list files for ivectors
p.addParamValue('Tnorm', '0',@ischar);

% p.addRequired('filename', @ischar);
p.parse(varargin{:});
disp('===================================================');
disp('Running iVector system scoring script for SRE2012.');
disp('===================================================');
disp('Input parameters:');
disp(p.Results);

%% Parse the method
ScoringMethod = regexp(p.Results.Scoring,'_','split');
DimReducMethod = regexp(p.Results.DimReduc,'_','split');
NormMethod = regexp(p.Results.Norm,'_','split');
Precision = p.Results.Precision;
RadialGaussianizationFlag = 0;

%% Todo for SRE12
% Be able to process trials with multiple training segments
% Average ivectors for each speaker before PLDA
% Use score level averaging

%% Required functions:
% load_binary_features
% textread
% LoadIvectorList
% lda_train
% LDA
% wccn_tran; Gaussian_PLDA; train_plda_model; score_plda_model; EvalSys

%% Load Ivectors
disp('Loading ivectors..');
iVecSize = str2double(p.Results.iVecSize);

if(strcmp(p.Results.ReadMode,'matrix'))
    iVectorDEVmat = load_binary_features(p.Results.iVecDev,iVecSize,Precision);
    iVectorTRAINmat = load_binary_features(p.Results.iVecTrain,iVecSize,Precision);
    iVectorTESTmat = load_binary_features(p.Results.iVecTest,iVecSize,Precision);
    if(~isempty(p.Results.iVecWCCN))
        iVectorWCCNmat = load_binary_features(p.Results.iVecWCCN,iVecSize,Precision);
        WCCNSpkrIDs = textread(p.Results.WCCNSpkrID)+1;
    end
elseif(strcmp(p.Results.ReadMode,'list'))
    
    iVectorDEVmat = LoadIvectorList(p.Results.iVecDev,iVecSize,Precision);
    iVectorTRAINmat = LoadIvectorList(p.Results.iVecTrain,iVecSize,Precision);
    iVectorTESTmat = LoadIvectorList(p.Results.iVecTest,iVecSize,Precision);
    if(~isempty(p.Results.iVecWCCN))
        iVectorWCCNmat = LoadIvectorList(p.Results.iVecWCCN,iVecSize,Precision);
        WCCNSpkrIDs = textread(p.Results.WCCNSpkrID)+1;
    end
end

%% Load lists
if(isempty(p.Results.TrialsIndex))
    [model, ~] = textread(p.Results.TrainList,'%d %s');
    [trial_model, trial_test_file, answers] = textread(p.Results.Trials,'%d %s %s');
    [test_file] = textread(p.Results.TestList,'%s');
else
    fid = fopen(p.Results.TrialsIndex);
    tline = fgetl(fid);
    TMP = regexp(tline,' ','split');
    fclose(fid);
    if(numel(TMP) == 5)
        disp('Processing SRE12 trials..');
        [TrainFileIDs,train_id,test_id,answers,KnownUnknown] = ProcessTrials(p.Results.TrialsIndex);
        iVectorTRAINmat_new = zeros(iVecSize,train_id(end));
        for i = 1:train_id(end)
            iVectorTRAINmat_new(:,i) = mean(iVectorTRAINmat(:,TrainFileIDs{i}),2);
        end
        
        iVectorTRAINmat = iVectorTRAINmat_new;
        clear iVectorTRAINmat_new;
    else
        disp('Processing SRE10 trials..');
        % Process the TrainiVector list:
        [train_id, test_id, answers] = textread(p.Results.TrialsIndex,'%d %d %d');
    end
end

%[answers] = textread(p.Results.Ans,'%s');
DevSpkrIDs = textread(p.Results.DevSpkrID)+1;
Ntrials = length(answers);
NDevData = size(iVectorDEVmat,2);
Nspkrs = max(DevSpkrIDs);
FileIndex=1:NDevData;

FinalScores = zeros(1,Ntrials);
fid_out = fopen(p.Results.OutScores,'w');

%% Using training speaker IDs in DEV
if(~isempty(p.Results.TrainSpkrID) && strcmp(p.Results.UseTrainForDEV,'1'))
    disp('Using training i-vectors for DEV');
    TrainSpkrIDs = textread(p.Results.TrainSpkrID);
    TrainSpkrIDs_new = zeros(length(TrainSpkrIDs),1);
    PreSpkrID = -1;
    CurrentSpkrID = DevSpkrIDs(end);
    for j = 1:length(TrainSpkrIDs)
        if(TrainSpkrIDs(j) ~= PreSpkrID)
            CurrentSpkrID = CurrentSpkrID+1;
        end
        TrainSpkrIDs_new(j) = CurrentSpkrID;
        PreSpkrID = TrainSpkrIDs(j);
    end
%     numel(DevSpkrIDs)
%     numel(TrainSpkrIDs_new);
    DevSpkrIDs = [DevSpkrIDs;TrainSpkrIDs_new(1:size(iVectorTRAINmat,2))];
    iVectorDEVmat = [iVectorDEVmat,iVectorTRAINmat];
    clear TrainSpkrIDs_new;
end

%% LDA_WCCN matrix training

% Nlda = 1:NDevData;
% Ndata_to_use=2718;
% Ndata_to_use=11748;
Ndata_to_use = NDevData;
Nlda  = 1:Ndata_to_use;
Nwccn = 1:Ndata_to_use;
% Nlda = 100;

for nn = 1:numel(NormMethod)
    if(strcmp(char(NormMethod(nn)),'MN'))
        disp('Performing MVN for i-Vectors..');
        m0 = mean(iVectorDEVmat,2);
        for k = 1:size(iVectorTRAINmat,2)
            iVectorTRAINmat(:,k) = iVectorTRAINmat(:,k) - m0;
        end
        for k = 1:size(iVectorTESTmat,2)
            iVectorTESTmat(:,k) = (iVectorTESTmat(:,k) - m0);
        end
        for k = 1:size(iVectorDEVmat,2)
            iVectorDEVmat(:,k) = (iVectorDEVmat(:,k) - m0);
        end
    end
    
    if(strcmp(char(NormMethod(nn)),'MVN'))
        disp('Performing MVN for i-Vectors..');
        m0 = mean(iVectorDEVmat,2);
        v0 = std(iVectorDEVmat,0,2);
        for k = 1:size(iVectorTRAINmat,2)
            iVectorTRAINmat(:,k) = (iVectorTRAINmat(:,k) - m0)./v0;
        end
        for k = 1:size(iVectorTESTmat,2)
            iVectorTESTmat(:,k) = (iVectorTESTmat(:,k) - m0)./v0;
        end
        for k = 1:size(iVectorDEVmat,2)
            iVectorDEVmat(:,k) = (iVectorDEVmat(:,k) - m0)./v0;
        end
    end
    
    if(strcmp(char(NormMethod(nn)),'Lnorm'))
        disp('Performing i-Vector standard length normalization..');
        for k = 1:size(iVectorTRAINmat,2)
            iVectorTRAINmat(:,k) = iVectorTRAINmat(:,k)./norm(iVectorTRAINmat(:,k));
        end
        for k = 1:size(iVectorTESTmat,2)
            iVectorTESTmat(:,k) = iVectorTESTmat(:,k)./norm(iVectorTESTmat(:,k));
        end
        for k = 1:size(iVectorDEVmat,2)
            iVectorDEVmat(:,k) = iVectorDEVmat(:,k)./norm(iVectorDEVmat(:,k));
        end
    end
    
    if(strcmp(char(NormMethod(nn)),'WT'))
        disp('Performing whitening..');
        SigmaX = cov(iVectorDEVmat');
        [U,D] = eig(SigmaX);
        WT = diag(1./sqrt(diag(D)))*U';
        iVectorTRAINmat = WT*iVectorTRAINmat;
        iVectorTESTmat = WT*iVectorTESTmat;
        iVectorDEVmat = WT*iVectorDEVmat;
    end
    
    if(strcmp(char(NormMethod(nn)),'DC'))
        disp('Performing whitening..');
        SigmaX = cov(iVectorDEVmat');
        [U,D] = eig(SigmaX);
        iVectorTRAINmat = U'*iVectorTRAINmat;
        iVectorTESTmat = U'*iVectorTESTmat;
        iVectorDEVmat = U'*iVectorDEVmat;
    end
    
    if(strcmp(char(NormMethod(nn)),'EFR'))
        disp('Performing EFR..');
        for iter = 1:5
            SigmaX = cov(iVectorDEVmat');
            m0 = mean(iVectorDEVmat,2);
            [U,D] = eig(SigmaX);
            %         WT = diag(1./sqrt(diag(D)))*U';
            WT = U*diag(sqrt(diag(D)));
            %         WT = sqrt(SigmaX);
            
            iVectorDEVmat = WT\bsxfun(@minus,iVectorDEVmat,m0);
            iVectorTRAINmat = WT\bsxfun(@minus,iVectorTRAINmat,m0);
            iVectorTESTmat = WT\bsxfun(@minus,iVectorTESTmat,m0);
            
            % length normalization without for loop
            iVectorDEVmat = bsxfun(@rdivide, iVectorDEVmat, sqrt(sum(iVectorDEVmat.^2)));
            iVectorTRAINmat = bsxfun(@rdivide, iVectorTRAINmat, sqrt(sum(iVectorTRAINmat.^2)));
            iVectorTESTmat = bsxfun(@rdivide, iVectorTESTmat, sqrt(sum(iVectorTESTmat.^2)));
        end
    end
end

if(numel(strfind(char(DimReducMethod(1)), 'LDAy')))
    disp('Training LDA using Yun''s matlab code ..');
    DIMLDA = str2double(char(DimReducMethod(2)));
    sprintf('Training LDA matrix. using %d data. LDA Dimension: %d\n',Nlda,DIMLDA);
    
    [WLDA, ~] = lda_train(iVectorDEVmat(:,Nlda),DevSpkrIDs(Nlda),FileIndex(Nlda),size(iVectorDEVmat,1));
    
    iVectorDEVmat = WLDA(:,1:DIMLDA)'*iVectorDEVmat;
    iVectorTRAINmat = WLDA(:,1:DIMLDA)'*iVectorTRAINmat;
    iVectorTESTmat = WLDA(:,1:DIMLDA)'*iVectorTESTmat;
    if(~isempty(p.Results.iVecWCCN))
        iVectorWCCNmat = WLDA(:,1:DIMLDA)'*iVectorWCCNmat;
    end
elseif(numel(strfind(char(DimReducMethod(1)), 'LDA')))
    disp('Training LDA using downloaded code ..');
    DIMLDA = str2double(char(DimReducMethod(2)));
    sprintf('Training LDA matrix. using %d data. LDA Dimension: %d\n',Nlda,DIMLDA);
    
    LDAoptions=[];
    LDAoptions.Regu = 1;
    %LDAoptions.PCARatio = 10;
    LDAoptions.Fisherface = 1;
    
    % Train LDA matrix
    [WLDA,~] = LDA(DevSpkrIDs,LDAoptions,iVectorDEVmat');
    
    iVectorDEVmat = WLDA(:,1:DIMLDA)'*iVectorDEVmat;
    iVectorTRAINmat = WLDA(:,1:DIMLDA)'*iVectorTRAINmat;
    iVectorTESTmat = WLDA(:,1:DIMLDA)'*iVectorTESTmat;
    if(~isempty(p.Results.iVecWCCN))
        iVectorWCCNmat = WLDA(:,1:DIMLDA)'*iVectorWCCNmat;
    end
else
    disp('No dimensionality reduction will be performed..');
end

if(cell2mat(strfind(NormMethod, 'WCCN')))
    disp('Training WCCN matrix..');
    % Train WCCN matrix
    if(~isempty(p.Results.iVecWCCN))
        Awccn = wccn_train(iVectorWCCNmat,WCCNSpkrIDs,[1:length(WCCNSpkrIDs)],size(iVectorDEVmat,1));
    else
        Awccn = wccn_train(iVectorDEVmat(:,Nwccn),DevSpkrIDs(Nwccn),FileIndex(Nwccn),size(iVectorDEVmat,1));
    end
    iVectorTRAINmat = Awccn'*iVectorTRAINmat;
    iVectorTESTmat = Awccn'*iVectorTESTmat;
end

if(cell2mat(strfind(NormMethod, 'RG')))
    RadialGaussianizationFlag = 1;
end

if(cell2mat(strfind(NormMethod, 'NONE')))
    disp('No normalization will be performed.');
end

%% PLDA
if(strcmp(char(ScoringMethod(1)), 'PLDA'))
    disp('Using Gaussian-PLDA model..');
    % preparation for Daniel's code
    % [scores,m,W,Phi,Sigma] = Gaussian_PLDA(dev,dev_spk_idx,mod,tst,nPhi[,flag_lnorm = 1]);
    FinalScores = zeros(1,Ntrials);
    [dev_spk_idx, ~] = hist(DevSpkrIDs,numel(unique(DevSpkrIDs)));
    nPhi = str2double(char(ScoringMethod(2)));
    [scores,~,~,~,~] = Gaussian_PLDA(iVectorDEVmat,dev_spk_idx',iVectorTRAINmat,iVectorTESTmat,nPhi,RadialGaussianizationFlag);
    
    if(strcmp(p.Results.Tnorm,'1'))
        disp('Performing T-norm');
         scores = bsxfun(@minus,scores,(mean(scores)));
         scores = bsxfun(@rdivide,scores,(var(scores)));
    end
    
    for i = 1:Ntrials
        FinalScores(i) = scores(train_id(i),test_id(i));
        if(answers(i) == 1)
            fprintf(fid_out,'%d %d %f %s\n',0,0,FinalScores(i),'target');
        elseif(answers(i) == 0)
            fprintf(fid_out,'%d %d %f %s\n',0,0,FinalScores(i),'nontarget');
        end
    end
elseif(strcmp(char(ScoringMethod(1)), 'EMPLDA'))
    disp('Using EM-PLDA model..');
    
    % This is Niko's code
    FinalScores = zeros(1,Ntrials);
    
    % usually nvoices and nchannels are 120
    
    nvoices = str2double(char(ScoringMethod(2)));
    if(numel(ScoringMethod) > 2); nchannels  = str2double(char(ScoringMethod(3))); else nchannels = nvoices;   end
    if(numel(ScoringMethod) > 3); niter = str2double(char(ScoringMethod(4))); else niter = 100;   end
    
    % generating speaker ID list in character
    DevSpkrIDs_char = textread(p.Results.DevSpkrID,'%s');
    % Training the model
    model = train_plda_model(iVectorDEVmat,DevSpkrIDs_char,nvoices,nchannels,niter);
    
    % testing: generating score matrix
    scores = score_plda_model(model,iVectorTRAINmat,iVectorTESTmat);
    
    for i = 1:Ntrials
        FinalScores(i) = scores(train_id(i),test_id(i));
        if(answers(i) == 1)
            fprintf(fid_out,'%d %d %f %s\n',0,0,FinalScores(i),'target');
        elseif(answers(i) == 0)
            fprintf(fid_out,'%d %d %f %s\n',0,0,FinalScores(i),'nontarget');
        end
    end
end


if(strcmp(char(ScoringMethod(1)), 'CD2'))
    disp('Using CD scoring..');
    FinalScores = zeros(1,Ntrials);
    
    % testing: generating score matrix
    scores = iVectorTRAINmat'*iVectorTESTmat;
    
    for i = 1:Ntrials
        FinalScores(i) = scores(train_id(i),test_id(i));
        if(answers(i) == 1)
            fprintf(fid_out,'%d %d %f %s\n',0,0,FinalScores(i),'target');
        elseif(answers(i) == 0)
            fprintf(fid_out,'%d %d %f %s\n',0,0,FinalScores(i),'nontarget');
        end
    end
end

%% Cosine distance scoring

disp('Scoring..');
if(strcmp(char(ScoringMethod(1)),'CD'))
    disp('Using cosine distance scoring');
elseif(strcmp(char(ScoringMethod(1)),'PLDA'))
    disp('Using PLDA scoring');
end

if(strcmp(char(ScoringMethod(1)),'CD'))
    for i=1:Ntrials
        if(isempty(p.Results.TrialsIndex))
            CurrentModel = trial_model(i);
            CurrentModel_id = find(model==CurrentModel);
            if(isempty(CurrentModel_id))
                disp(['Couldnt find model : ',num2str(CurrentModel)]);
                return;
            end
            
            [~,TestFileBasename] = fileparts(char(trial_test_file(i)));
            TestFileIndex = find(cellfun('isempty',(strfind(test_file,TestFileBasename)))==0);
            if(isempty(TestFileIndex))
                disp(['Couldnt find test file :',TestFileBasename]);
                return;
            end
        else
            CurrentModel_id = train_id(i);
            TestFileIndex = test_id(i);
        end
        
        if(strcmp(char(ScoringMethod(1)),'CD'))
            % using cosine distance scoring
            TrainIvector = iVectorTRAINmat(:,CurrentModel_id);
            TestIvector = iVectorTESTmat(:,TestFileIndex(1));
            
            if(numel(TrainIvector) ~= numel(TestIvector))
                disp(['Error in Trial ',num2str(i)]);
                numel(TrainIvector)
                numel(TestIvector)
                return;
            end
            FinalScores(i) = sum(TrainIvector.*TestIvector)/(norm(TrainIvector)*norm(TestIvector));
            if(isnan(FinalScores(i)))
                1;
            end
            
        elseif(strcmp(char(ScoringMethod(1)),'PLDA'))
            % using PLDA scoring
            TrainIvector = iVectorTRAINmat(:,CurrentModel_id);
            TestIvector = iVectorTESTmat(:,TestFileIndex);
            FinalScores(i) = PLDA_Verification(PLDAModelSRE,TrainIvector,TestIvector);
        end
        
        if(isempty(p.Results.TrialsIndex))
            fprintf(fid_out,'%d %s %f %s\n',CurrentModel,TestFileBasename,FinalScores(i),char(answers(i)));
        else
            if(answers(i) == 1)
                fprintf(fid_out,'%d %d %f %s\n',CurrentModel_id,TestFileIndex,FinalScores(i),'target');
            elseif(answers(i) == 0)
                fprintf(fid_out,'%d %d %f %s\n',CurrentModel_id,TestFileIndex,FinalScores(i),'nontarget');
            end
        end
    end
end

if(~isempty(p.Results.TrialsIndex))
    % Usually use Rocchdet if trial index is provided.
    disp('Using ROCCHDET');
    [eer,mindcf_old,mindcf_new] = EvalSys(FinalScores,answers);
    if(~ isempty(p.Results.ResultFile))
        fid_res = fopen(p.Results.ResultFile,'w');
        %         fprintf(fid_res, 'DimReduc %s Norm %s EER %0.5f MinDCF_old %0.5f MinDCF_new %0.5f',p.Results.DimReduc, p.Results.Norm, eer,mindcf_old,mindcf_new);
        fprintf(fid_res, 'EER %0.5f MinDCF_old %0.5f MinDCF_new %0.5f\n',eer,mindcf_old,mindcf_new);
    end
else
    % If trial index is not provided use older tool for EER.
    system(['EvalSys ',p.Results.OutScores]);
end

function iVectorMat = LoadIvectorList(IvectorListFile,iVecSize,Precision)

disp(['iVector list: ',IvectorListFile]);
IvectorList = textread(IvectorListFile,'%s');
Nfiles = length(IvectorList);
iVectorMat = zeros(iVecSize,Nfiles);
for i=1:Nfiles
    iVectorMat(:,i) = load_binary_features(char(IvectorList(i)),iVecSize,Precision);
end
