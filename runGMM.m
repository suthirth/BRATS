info = mha_read_header('BRATS_HG0002_T1C.mha');
X = mha_read_volume(info);
imshow(squeeze(X(:,:,88)),[]);

%Prepare Mask

model = BuildGMM(trainData, truthData);