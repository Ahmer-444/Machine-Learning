clear all
clc
close all

images_dir = 'sidefeet/';
create_dataset(images_dir,[64 64]);

% First 100 Thumnails to view the training dataset
load('SideFootData.mat');
figure;
thumbnails = data_array(:,:,:,1:100);
montage(thumbnails)