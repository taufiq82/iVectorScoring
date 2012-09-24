function W = wccn_train(Eigenvoice,IdxSpk,IdxSession,NEV)

%------------------------------
%load speaker eigenbasis
NumSpk=max(IdxSpk);
NumSession=max(IdxSession);

EigenvoiceSpk = zeros(NEV, NumSpk);
NumSequence = zeros(NumSpk,1);
for i=1:NumSession
    EigenvoiceSpk(:,IdxSpk(i)) = EigenvoiceSpk(:,IdxSpk(i)) + Eigenvoice(:,IdxSession(i));
    NumSequence(IdxSpk(i)) = NumSequence(IdxSpk(i)) + 1;
end
for i=1:NumSpk
    EigenvoiceSpk(:,i) = EigenvoiceSpk(:,i)/NumSequence(i);
end
tmpSwithin=zeros(NEV,NEV,NumSpk);

DiffWithin = zeros(NEV,NumSession);
for i=1:NumSession
   DiffWithin(:,i)=Eigenvoice(:,i)-EigenvoiceSpk(:,IdxSpk(i));
   tmpSwithin(:,:,IdxSpk(i)) = tmpSwithin(:,:,IdxSpk(i)) + DiffWithin(:,i)*DiffWithin(:,i)';
end
for i=1:NumSpk
    tmpSwithin(:,:,i) = tmpSwithin(:,:,i)/NumSequence(i);
end
Swithin=zeros(NEV,NEV);
for i=1:NumSpk
    Swithin = Swithin + tmpSwithin(:,:,i);
end
Swithin = Swithin/NumSpk;
R = inv(Swithin);
W = chol(R, 'lower');