function [TrainFileIDs,TrainSpkrID,TestFileID,Answers,KnownUnknown] = ProcessTrials(InFile)
% function [TrainFileIDs,TrainSpkrID,TestFileID,Answers,KnownUnknown] = ProcessTrials(InFile,OutFile)

fid = fopen(InFile);
DATA = textscan(fid,'%d %s %d %d %d');

Nspkrs = DATA{1}(end);

% Training File IDs for a speaker pin 
TrainFileIDs = cell(1,Nspkrs);

TrainSpkrID = DATA{1};
TestFileID = DATA{3};
Answers = DATA{4};
KnownUnknown = DATA{5};

for SpkrNo = 1:Nspkrs
    TempIndex = find(DATA{1} == SpkrNo);
    TMP = char(DATA{2}(TempIndex(1)));
    TMP = TMP(1:end-1);
    TrainFileIDs{SpkrNo} = str2num(char(regexp(TMP,',', 'split')))';
end

% save(OutFile,'TrainFileIDs','TrainSpkrID', 'TestFileID','Answers','KnownUnknown');

