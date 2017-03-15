

mri = MRIread('1119_3.nii');mri = mri.vol;
% true_label_file = importdata('./1119_3_glm.mat');
% cnn_pred_label_file = importdata('./1119_6patches.mat');cnn_pred_label_file = cnn_pred_label_file.label;
% reg_pred_label_file = importdata('./pred_test_regAll/label/1119_3_glm.mat'); reg_pred_label_file = reg_pred_label_file.label;
% 
% 
% cnn_pred_label_file = convert2MiccaiLabels(cnn_pred_label_file);
% reg_pred_label_file = convert2MiccaiLabels(reg_pred_label_file);

aSag_pred_cnn = squeeze(cnn_pred_label_file(100,:,:));
aSag_pred_reg = squeeze(reg_pred_label_file(100,:,:));
aSag_pred_true = squeeze(true_label_file(100,:,:));

aSag_diff_cnn = double(aSag_pred_cnn)-double(aSag_pred_true);
aSag_diff_reg = double(aSag_pred_reg)-double(aSag_pred_true);

imagesc(aSag_diff_cnn);title('cnn')
figure
imagesc(aSag_diff_reg);title('reg')


I = squeeze(mri(100,:,:));

close all
imshow((I),[]);
hold on
h=imshow(aSag_pred_true);
hold off
set(h,'AlphaData',I)



viewMaskedVolume(mri, true_label_file, 'Orig');

viewMaskedVolume(mri, cnn_pred_label_file, 'cnn');

viewMaskedVolume(mri, reg_pred_label_file, 'reg');

viewMaskedVolume(mri, double(cnn_pred_label_file)-double(true_label_file), 'diff_cnn');

viewMaskedVolume(mri, double(reg_pred_label_file)-double(true_label_file), 'diff_reg');