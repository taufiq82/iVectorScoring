function feature_matrix=load_binary_features(filename,feature_dimension,data_type)
% feature_matrix=load_binary_features(filename,feature_dimension,data_type)
if nargin<3
    data_type='short';
end
% feature_fid=fopen(filename,'wb');
% filename='/localdisk/Taufiq/Research/YOHO_MFCC_13_voiced/DEV/180/1/26_81_57.hhtc';
% filename='/localdisk/ubm_basic/Yoho_data_1/dev_features/180/1/67_76_24.mfcc'
% feature_dimension=38;
%%
fid=fopen(filename,'r');
% feature_matrix=fread(fid,[feature_dimension,Inf],'long');
% feature_matrix=fread(fid,[feature_dimension,Inf],'double');
feature_matrix=fread(fid,[feature_dimension,Inf],data_type);
fclose(fid);
% surf(feature_matrix);
