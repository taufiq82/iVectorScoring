%
% function LogLikeRatio = PLDA_Verification(Model, Data1, Data2)
%
% Face verification using PLDA based on model comparison
%   Calculate the ratio of loglikelihood of the model that two faces Data1
%   and Data2 match to that of two faces do not match:
% 
%       LogLikeRatio = Loglike_Match / Loglike_NotMatch
%
%   Inputs:
%       Model   - Learned PLDA model
%       Data1   - Data point 1: NFeature x 1
%       Data2   - Data point 2: NFeature x 1
%   Outputs:
%       LogLikeRatio -  Scalar 
%
%******************** Disclaimer *****************************************
%*** This program is distributed in the hope that it will be useful, but
%*** WITHOUT ANY WARRANTY; without even the implied warranty of 
%*** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%*** Feel free to use this code for academic purposes.  Plase use the
%*** citation provided below.
%
% S.J.D. Prince and J.H. Elder, “Probabilistic linear discriminant analysis
% for inferences about identity,”  ICCV, 2007. 
%
% P. Li and S.J.D. Prince, “Probabilistic Methods for Face Registration and
% Recognition”, In Advances in Face Image Analysis: Techniques and
% Technologies, Y. Zhang (eds.)  (in press). 
%
%**************************************************************************
%
% 04-03-2010
%
function LogLikeRatio = PLDA_Verification(Model, Data1, Data2)

% Get the learned model parameters
F = Model.F;
G = Model.G;
Sigma = Model.Sigma;
meanVec = Model.meanVec;

% Subtract mean from test data
Data1 = Data1 - meanVec;
Data2 = Data2 - meanVec;

% Preprocess PLDA model and data to accelerate the computation
HIGHEST_N = 2; % highest number of faces explained by one variable
factorModel = preProcessPLDAModel(F, G, Sigma, HIGHEST_N);
Data1P = preProcessPLDAData(factorModel, Data1);
Data2P = preProcessPLDAData(factorModel, Data2);

% Calculate the log likelihood of the model that two images do not match
logLikeNoMatch = getLogLikeMatchPLDA(factorModel, Data1P)...
    + getLogLikeMatchPLDA(factorModel, Data2P);

% Calculate the log likelihood of the model that two images match
logLikeMatch = getLogLikeMatchPLDA(factorModel, [Data1P, Data2P]);

LogLikeRatio = logLikeMatch - logLikeNoMatch;
% End of main funcion PLDA_Verification

%==========================================================================
% Preprocess the PLDA model to accelerate the calculation
function factorModel = preProcessPLDAModel(F, G, Sigma, HIGHEST_N)

[N_DATA_DIM N_HIDDEN_DIM] = size(F);
[N_DATA_DIM N_HIDDEN_NOISE_DIM] = size(G);

invCovDiag = 1./Sigma;
GWeighted = G.*repmat(invCovDiag,1,N_HIDDEN_NOISE_DIM); % \Simga^(-1) * G

% (G' * \Simga^(-1) * G + I) ^ (-1)
factorModel.invTerm = inv(G'*GWeighted+eye(N_HIDDEN_NOISE_DIM));

% (F' * \Simga^(-1) - F' *  \Simga^(-1) * G * (G' * \Simga^(-1) * G + I) ^ (-1) * G' * \Simga^(-1)
% = F' * (G * G' + \Sigma)^(-1)
factorModel.FTranspJ = (F.*repmat(invCovDiag,1,N_HIDDEN_DIM))'-(F'*GWeighted)*factorModel.invTerm*GWeighted';

%using matrix determinant lemma http://www.ee.ic.ac.uk/hp/staff/www/matrix/identity.html
factorModel.AInv = invCovDiag;      % \Sigma^(-1)
factorModel.U = -1*GWeighted;       % - \Simga^(-1) * G

%(G' * \Simga^(-1) * G + I) ^ (-1) * G' * \Simga^(-1)
factorModel.V = inv(G'*GWeighted+eye(N_HIDDEN_NOISE_DIM))*GWeighted';
%  I - (G' * \Simga^(-1) * G + I) ^ (-1) * G' * \Simga^(-1) * G
factorModel.MDLInvTerm = eye(N_HIDDEN_NOISE_DIM)+(factorModel.V*(repmat(Sigma,1,N_HIDDEN_NOISE_DIM).*factorModel.U));

% log(det((G * G' +  \Sigma)^(-1))
[U L V]=svd(factorModel.MDLInvTerm);
factorModel.logDetJ =sum(log(diag(L))) + sum(log(invCovDiag));

factorModel.F = F;
factorModel.G = G;

% (G * G' + \Sigma)^(-1) * F
factorModel.FWeighted = factorModel.FTranspJ';
% \Simga^(-1) * G
factorModel.GWeighted = GWeighted;

for cN = 1 : HIGHEST_N
    [U, L, V] = svd(cN*F'*factorModel.FWeighted+eye(N_HIDDEN_DIM));
    DiagL = diag(L);
    factorModel.invNFSFPlusIDiag{cN} = V * diag(1 ./ DiagL) * U';    
    logdetInvNFSFPlusIDiag = sum(log(1 ./ DiagL));    
    
    factorModel.constTerm{cN} = - (cN * N_DATA_DIM / 2) * log(2* pi)...
        + cN / 2 * factorModel.logDetJ + 0.5 * logdetInvNFSFPlusIDiag;
end
% End of function preProcessPLDAModel

%==========================================================================
% Preprocess the data to acclerate the computation
function dataPP = preProcessPLDAData(factorModel, data)
[N_DATA_DIM N_DATA] = size(data);
for cData = 1 : N_DATA
    dataPP(cData).FTinvSx = factorModel.FTranspJ*data(:,cData);
    % Calculate quadratic term for log Gaussian probability
    quadTerm1 = (factorModel.AInv.*data(:,cData))'*data(:,cData);
    quadTerm2 = (data(:,cData)'*factorModel.GWeighted)*factorModel.invTerm*(factorModel.GWeighted'*data(:,cData));
    quadTerm = quadTerm1-quadTerm2;
    
    dataPP(cData).quadTerm = quadTerm;    
end
% End of function preProcessPLDAData


%==========================================================================
% Calculate the likelihood of model that the data in dataPP are from the
% same identity.
function logLike = getLogLikeMatchPLDA(factorModel, dataPP)

N_DATA = length(dataPP);
[N_Original, N_HIDDEN] = size(factorModel.F);

logLike = factorModel.constTerm{N_DATA};

sumWeightedData = zeros(N_HIDDEN, 1);
logTerm = 0;
for cData = 1 : N_DATA
    logTerm = logTerm + dataPP(cData).quadTerm;        
    sumWeightedData = sumWeightedData + dataPP(cData).FTinvSx;
end
logLike = logLike - 0.5 *(logTerm - ...
    (sumWeightedData' * factorModel.invNFSFPlusIDiag{N_DATA} * sumWeightedData));
return
% End of function preProcessPLDAModel