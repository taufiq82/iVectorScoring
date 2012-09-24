%******************** Disclaimer *****************************************
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%
% The program is free for academic use. If you are interested in using the
% software for commercial purposes.  Please contact
%   Dr. Simon J. D. Prince
%   Email: s.prince@cs.ucl.ac.uk
%
% Plase use the citation provided below if it is useful to your research:
%
% S.J.D. Prince and J.H. Elder, �Probabilistic linear discriminant analysis
% for inferences about identity,�  ICCV, 2007.
%
% P. Li and S.J.D. Prince, �Probabilistic Methods for Face Registration and
% Recognition�, In Advances in Face Image Analysis: Techniques and
% Technologies, Y. Zhang (eds.)  (in press).
%
%**************************************************************************
%   function Model = PLDA_Train(Data, ImageID, N_ITER, N_F, N_G)
%
%   Learn a PLDA model given training Data using EM algorithm. As a specific
%   form of facor analysis model, there is no unique solution to PLDA. So
%   a different set of parameters will be obtained for each run.
%
%   Input:
%       Data    - NFeature  x NSample   Training data
%       ImageID - NSample   x nIdentity Identity matrix of training data
%       N_ITER  - Number of iteration. The convergence of the EM algorithm
%                 is quite fast. We use 6 iterations in the experiments in
%                 the ICCV paper. But it may takes more iterations (20~50).
%
%       N_F     - Number of Factors for H(Hiden dimension)
%       N_G     - Number of Factors for W(Hiden dimension)
%                 We keep the hiden subspace dimensions of identity variable
%                 H and noise varialbe W the same, though this may not be
%                 necessary. This number should be smaller than the number
%                 of identities (nIdentity) and observed space dimension
%                 (NFeature) of the training data.
%   Output:
%       Model - Learned PLDA model with the following parameters
%           F        - NFeature  x NFeature_H Factor loading matrix
%           G        - NFeature  x NFeature_W Factor loading matrix
%           Sigma    - NFeature  x 1 Covariance matrix (diagonal)
%           meanVec  - NFeature  x 1 Mean vector of the training data
%
%   To make this work well, read the note below about rescaling the
%   covariance!
%**************************************************************************
%   04-03-2010 Version 1
function Model = PLDA_Train(Data, ImageID, N_ITER, N_F, N_G)

% Remove mean
meanVec = mean(Data, 2);
N_DATA = size(Data, 2);
Data = double(Data) - repmat(meanVec, 1, N_DATA);

% Initialize the parameters randomly
OBS_DIM = size(Data, 1); % Dimension of original space
G = randn(OBS_DIM, N_G);
Sigma = 0.01 .* ones(OBS_DIM, 1);

% Initialize factor loading F to between space to initialize
clusterMeans = (Data * ImageID) * (diag(1 ./ (sum(ImageID))));
F =  trainPCA(clusterMeans);
F = F(:, 1:N_F);

% EM algorithm
for cIter = 1 : N_ITER
    % E - Step: Calculate expected value of h and hh (summation)
    [Eh EhhSum] = getExpectedValuesPLDA(F, G, Sigma, Data, ImageID);
    
    % M - Step: Updating the parameters
    
    % Calculate terms for updates
    xhSum = zeros(size(Data,1), size(Eh, 1));
    for cData = 1 : N_DATA
        xhSum = xhSum + Data(:, cData) * Eh(:, cData)';
    end;
    
    % Update factor loading matrix [F G]
    FGEst = xhSum * inv(EhhSum) ;
    
    % Update covarance matrix of the noise Sigma
    Sigma = mean(Data .* Data - (FGEst * Eh) .* Data,2);
    
    % Extract F and G separateley
    F = FGEst(:, 1 : N_F);
    G = FGEst(:, N_F + 1 : end);
end

% Deal with scale issues - deals with slow convergence to final answer.
% There is a nasty hack here. For high dimensional data, we find
% empirically that multiplying the covariance matrix Sigma by a fixed
% amount improves the performance.
% We have never understood the reason for this (and we're not totally sure
% it's not becuase of a but somewhere) so we didn't discuss it in the ICCV
% paper.  For what it's worth however, we used a scaling factor of 100 for
% the 14700 dimensional XM2VTS pixel data. To activate this,  one may set
% Rescale =  1 and change the scaling (100).
Rescale = 0;
if Rescale
    EhSD = sqrt(diag(cov(Eh')));
    FGEst = FGEst .* repmat(EhSD',OBS_DIM,1);
    % Extract F and G separateley
    F = FGEst(:,1:N_F);
    G = FGEst(:,N_F+1:end);
    Sigma = Sigma * 100;
end

% Store the trained PLDA model
Model.F = F;
Model.G = G;
Model.Sigma = Sigma;
Model.meanVec = meanVec;

%=======================================================================
% function [Eh,EhhSum] = getExpectedValuesPLDA(F, G, Sigma, x, ImageID)
%
%   E(Expectation ) - step of EM algorithm
%
%   Input:
%       F       - Factor loading matrix of latent identity variable
%       G       - Factor loading matrix of latent noise variable
%       Sigma   - Covariance matrix of noise (diagonal)
%       x       - Training data (zero mean)
%       ImageID - Identity matrix of training data
%
%   Output:
%       Eh      - Expectation of latent varialbe h and latent noise
%                 variable w: (NFeature_H + NFeature_W) x NSample matrix
%       EhhSumm - Summation of the expectation of covariances of latent
%                 variables:
%                 (NFeature_H + NFeature_W) x (NFeature_H + NFeature_W)
function [Eh,EhhSum] = getExpectedValuesPLDA(F,G,Sigma,x,ImageID)

N_HID_DIM = size(F, 2);
N_HID_DIM_NOISE = size(G, 2);
N_DATA = size(x,2);
N_INDIV = size(ImageID,2);

% Create space for output data
Eh = zeros(N_HID_DIM+N_HID_DIM_NOISE,N_DATA);
EhhSum  = zeros(N_HID_DIM+N_HID_DIM_NOISE,N_HID_DIM+N_HID_DIM_NOISE);


% Calculate all inverse terms in advance to save computation
repeatValues = unique(sum(ImageID));
nRepeatValues = length(repeatValues);
invTermsAll = cell(nRepeatValues, 1);
for cRepeatVal = 1 : nRepeatValues
    thisRepVal = repeatValues(cRepeatVal);
    %create A matrix and sigmaMatrix
    ATISigA = zeros(N_HID_DIM+thisRepVal*N_HID_DIM_NOISE);
    weightedF = F.*(repmat(1./Sigma,1,N_HID_DIM));
    weightedG = G.*(repmat(1./Sigma,1,N_HID_DIM_NOISE));
    
    ATISigA(1:N_HID_DIM,1:N_HID_DIM) = thisRepVal*weightedF'*F;
    for cMat = 1:thisRepVal
        ATISigA(N_HID_DIM+(cMat-1)*N_HID_DIM_NOISE+1:N_HID_DIM+cMat*N_HID_DIM_NOISE,1:N_HID_DIM) = weightedG'*F;
        ATISigA(1:N_HID_DIM,N_HID_DIM+(cMat-1)*N_HID_DIM_NOISE+1:N_HID_DIM+cMat*N_HID_DIM_NOISE) = weightedF'*G;
        ATISigA(N_HID_DIM+(cMat-1)*N_HID_DIM_NOISE+1:N_HID_DIM+cMat*N_HID_DIM_NOISE,...
            N_HID_DIM+(cMat-1)*N_HID_DIM_NOISE+1:N_HID_DIM+cMat*N_HID_DIM_NOISE) = weightedG'*G;
    end;
    invTerm =  inv(eye(N_HID_DIM+thisRepVal*N_HID_DIM_NOISE)+ATISigA);
    
    %store all of these values
    invTermsAll{repeatValues(cRepeatVal)} = invTerm;
end

% Run through each individual
for cInd = 1 : N_INDIV
    % Figure out how many data points we are combining here
    nFaces = full(sum(ImageID(:,cInd)));
    
    % Image indices
    thisImIndex = find(ImageID(:,cInd));
    
    % Concatenate data from this individual
    dataAll = x(:,thisImIndex).*repmat(1./Sigma,1,nFaces);
    
    % Extract relevant terms
    ATISigX = zeros(N_HID_DIM+nFaces*N_HID_DIM_NOISE,1);
    ATISigX(1:N_HID_DIM,:) = sum(F'*dataAll,2);
    for cIm = 1 : nFaces
        ATISigX(N_HID_DIM+(cIm-1)*N_HID_DIM_NOISE+1:N_HID_DIM+cIm*N_HID_DIM_NOISE,:) = G'*dataAll(:,cIm);
    end;
    
    invTerm = invTermsAll{nFaces};
    
    % Combine appropriately
    thisEh = invTerm*ATISigX;
    thisEhh = invTerm+thisEh*thisEh';
    
    % Extract each different part for the final Eh components
    for cFaces = 1 : nFaces
        thisIndex = [1:N_HID_DIM  N_HID_DIM+(cFaces-1)*N_HID_DIM_NOISE+1:N_HID_DIM+cFaces*N_HID_DIM_NOISE];
        Eh(:,thisImIndex(cFaces)) = thisEh(thisIndex);
        EhhSum=EhhSum+thisEhh(thisIndex,thisIndex);
    end
end
% End of function getExpectedValuesPLDA

%=======================================================================
% function  [princComp meanVec] = trainPCA(data)
%
% Principla Component Analysis (PCA)
%
%   Input:
%       Data    - NFeature  x NSample   Training data
%
%   Output:
%       princComp   - Principal Expectation of latent varialbe h and latent
%                       noise 
%       meanVec     - Mean vector of the data
%
function [princComp meanVec] = trainPCA(data)
[~,nData] = size(data);
meanVec = mean(data,2);
data = data-repmat(meanVec,1,nData);

XXT = data'*data;
[~, LSq, V] = svd(XXT);
LInv = 1./sqrt(diag(LSq));
princComp  = data * V * diag(LInv);
% End of function trainPCA
