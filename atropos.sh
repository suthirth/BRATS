bash antsRegistrationSyN.sh -d 3 -f ~/BRATS_HG0001/BRATS_HG0001_T1.mha -m ~/Kirby/S_template3_BrainCerebellum.nii.gz -o ~/ANTs-Exp/MMRR-to-HG1 -t b -n 12
./antsApplyTransforms -d 3 -i ~/Kirby/Priors/priors1.nii.gz -o ~/ANTs-Exp/BRATS_HG001/prior1.nii.gz -r ~/BRATS_HG0001/BRATS_HG0001_T1.mha -t ~/ANTs-Exp/MMRR-to-HG11Warp.nii.gz -t ~/ANTs-Exp/MMRR-to-HG10GenericAffine.mat -n BSpline
python normalise_single.py
bash antsAtroposN4.sh -d 3 -a ~/BRATS_HG0001/BRATS_HG0001_T1.mha -x ~/ANTs-Exp/BRATS_HG001/mask.mha -c 3 -p ~/ANTs-Exp/BRATS_HG001/prior%d.nii.gz -o ~/ANTs-Exp/BRATS_HG001/atropos
