function model = BuildGMM(trainData)
%%
%Inputs:
%trainData: col(1) Intensity Values col(2) Label (1/2/3/4/0/8)
%Labels: 
%1 - Necrosis 2 - Surrounding Edema 3 - Non Enhancing Tumor 4 - Enhancing Tumor
%5 - White Matter 6 - Gray Matter 7 - CSF
%8 - Non-Brain
%Output:
%model: (cols) weight, mean, std devn (rows) class 1,2,3..

%% 
%Initialize cluster centres for all classses
%Use truth table to estimate intial for classes 1-4 
for i=1:2%4
    id = find(trainData(:,2) == i);
    labelData = double(trainData(id,1));
    CC_tumor(i,1) = mean(labelData);
    CC_tumor(i,2) = var(labelData);
    CC_tumor(i,3) = size(id,1)/size(find(trainData(:,2) ~= 8),1);
end

%K-Means to estimate initial for classes 5-7 
classData = double(trainData(find(trainData(:,2) == 0),1));
[IDX,CC_normal] = kmeans(classData, 3, 'Start', 'uniform', 'Display','iter'); 
for i=1:3
    id = find(IDX(:,1) == i);
    labelData = double(classData(id));
    CC_normal(i,2) = var(labelData);
    CC_normal(i,3) = size(id,1)/size(find(trainData(:,2) ~= 8),1);
end

model = [CC_tumor; CC_normal]

%%
%Expectation Maximisation to calculate final GMM parameters

%Initialization
brainData = double(trainData(find(trainData(:,2)~=8),1));
gamma = zeros(size(brainData,1),5);
J = zeros(size(brainData,1),5);
for i=1:size(brainData,1)
    for j=1:5%7
        J(i,j) = model(j,3) * ((2*pi)^(-0.5)) * (model(j,2)^(-0.5)) * exp(-((brainData(i)-model(j,1))^2)/(2*model(j,2)));
    end
end
j_new = sum(log(sum(J,2)),1)
j_old = j_new - 1000000

%EM iterations
while j_new - j_old > abs(10^-5 * j_old)
    j_old = j_new;
    %E Step
    for i=1:size(brainData,1)
        for j=1:5%7
            gamma(i,j) = model(j,3) * ((2*pi)^(-0.5)) * (model(j,2)^-0.5) * exp(-((brainData(i)-model(j,1))^2)/(2*model(j,2)));
        end
        gamma(i,:) = gamma(i,:)./sum(gamma(i,:));
    end
    
    %M Step
    for j =1:5%7
            model(j,1) = sum(gamma(:,j).*brainData)/sum(gamma(:,j));
            model(j,2) = sum(gamma(:,j).*((brainData - model(j,1)).^2))/sum(gamma(:,j));
            model(j,3) = sum(gamma(:,j))/size(brainData,1);
    end
    
    %Log Likelihood
    for i=1:size(brainData,1)
        for j=1:5%7
            J(i,j) = model(j,3) * ((2*pi)^(-0.5)) * (model(j,2)^-0.5) * exp(-((brainData(i)-model(j,1))^2)/(2*model(j,2)));
        end
    end
    j_new = sum(log(sum(J,2)),1)
    model

end

end
                        



    


