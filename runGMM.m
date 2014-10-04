%%
%Read all Training Data. eg- T1C HG files
trainData = [];
for i=1:2%7
    filename = sprintf('BRATS_HG%.4d_T1.mha',i);
    info = mha_read_header(filename);
    intensityValues = mha_read_volume(info);
    truthFile = sprintf('BRATS_HG%.4d_truth.mha',i);
    info = mha_read_header(truthFile);
    truthValues = mha_read_volume(info);
    dim = size(intensityValues);
    trainData = [trainData; reshape(intensityValues,dim(1)*dim(2)*dim(3),1) reshape(truthValues,dim(1)*dim(2)*dim(3),1)];
end

%%
%Label NAs as 8 (Non-Brain)
x = size(trainData,1);
for i=1:x
    if(trainData(i,1) == 0)
        trainData(i,2) = 8;
    end
end

%Display Histogram
%classData = trainData(find(trainData(:,2) == 0),1);
%hist(double(classData),50);

%Build GMM
model = BuildGMM(trainData);
       
%plot(squeeze(X(:,108,88)),[]);

%Prepare Mask



%model = BuildGMM(trainData, truthData);