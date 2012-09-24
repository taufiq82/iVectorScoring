function [A, D] = lda_train(Eigenvoice,IdxSpk,IdxSession,NEV)

%------------------------------
%load speaker eigenbasis
NumSpk=max(IdxSpk);
NumSession=max(IdxSession);

EigenvoiceSpk = zeros(NEV,NumSpk);
NumSequence = zeros(NumSpk,1);
for i=1:NumSession
    EigenvoiceSpk(:,IdxSpk(i)) = EigenvoiceSpk(:,IdxSpk(i)) + Eigenvoice(:,IdxSession(i));
    NumSequence(IdxSpk(i)) = NumSequence(IdxSpk(i)) + 1;
end
for i=1:NumSpk
    EigenvoiceSpk(:,i) = EigenvoiceSpk(:,i)/NumSequence(i);
end
EigenvoiceALL = sum(Eigenvoice, 2)/NumSession;

Sbetween = zeros(NEV,NEV);
% DiffBetween=zeros(NEV,NumSpk);
for i=1:NumSpk
%     DiffBetween(:,i)=EigenvoiceSpk(:,i)-EigenvoiceALL;
%     Sbetween = Sbetween + DiffBetween(:,i)*DiffBetween(:,i)';
      Sbetween = Sbetween + (EigenvoiceSpk(:,i)-EigenvoiceALL)*(EigenvoiceSpk(:,i)-EigenvoiceALL)';
end
tmpSwithin=zeros(NEV,NEV,NumSpk);

% DiffWithin = zeros(NEV,NumSpk);
for i=1:NumSession
%    DiffWithin(:,i)=Eigenvoice(:,i)-EigenvoiceSpk(:,IdxSpk(i));
%    tmpSwithin(:,:,IdxSpk(i)) = tmpSwithin(:,:,IdxSpk(i)) + DiffWithin(:,i)*DiffWithin(:,i)';
    tmpSwithin(:,:,IdxSpk(i)) = tmpSwithin(:,:,IdxSpk(i)) + (Eigenvoice(:,i)-EigenvoiceSpk(:,IdxSpk(i)))*(Eigenvoice(:,i)-EigenvoiceSpk(:,IdxSpk(i)))';
end

% for i=1:NumSpk
%     tmpSwithin(:,:,i) = tmpSwithin(:,:,i)/NumSequence(i);
% end

Swithin=zeros(NEV,NEV);
for i=1:NumSpk
    Swithin = Swithin + tmpSwithin(:,:,i)/NumSequence(i);
end
% M=inv(Swithin)*Sbetween;
M=Swithin\Sbetween;
[A, D] = eig(M);
[x,index]=sort(diag(D));
x=flipud(x);
index=flipud(index);
A=A(:,index);
D=diag(x);