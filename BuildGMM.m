function model = BuildGMM(trainData, truthData)
%Labels:
%Labeled from Training Data: 
%1 - Necrosis 
%2 - Surrounding Edema
%3 - Non Enhancing Tumor
%4 - Enhancing Tumor
%Unsupervised Labeling
%5 - White Matter
%6 - Gray Matter
%7 - CSF

%Initialize, resize trainData and all that

x = size(trainData,1);
y = size(trainData,2);
z = size(trainData,3);

if (size(truthData) ~= size(trainData))
    fprintf('Size mismatch');
end

classData = zeros(x*y*z,7);
count = zeros(1,7);

for i=1:x
    for j=1:y
        for k=1:z
            if(truthData(x,y,z) ~= 0)
                truthData(x,y,z) = 5; %Transfer all zero (unlabeled) to 5 (arbitrarily chosen)
            end
            count(truthData(x,y,z)) = count(truthData(x,y,z)) + 1;
            classData(count(truthData(x,y,z)),truthData(x,y,z)) = trainData(x,y,z);
        end
    end
end

%%
%K-Means to calculate cluster centres before GMM






                        



    


