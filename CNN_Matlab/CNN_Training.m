clear all
close all
clc

% If you want train the system on your custom data, Set "doTraining=True".
% If you want to load pretrained mode in order to calculate Validation Accuracy, Unset "doTraining=False".
doTraining = false;

% Load CIFAR Data or any Customize Data  
data = load('CIFAR-10.mat');

% Num of Object Categories = 10
NumofCategories = 10;

TrainingImages = data.trnImage;
ValidationImages = data.tstImage;
TrainingLabels = categorical(data.trnLabel);
ValidationLabels = categorical(data.tstLabel);

% Display first 100 Thumnails to view the training dataset
figure;
thumbnails = TrainingImages(:,:,:,1:100);
montage(thumbnails)

% Get the dimensions of your training data 
[height, width, depth, ~] = size(TrainingImages);
imageSize = [height width depth];

 % Set Convolutional Layer Parameters
filterSize = [5 5];
numFilters = 32;

% Initiallize CNN Network Architecture - see  CNN_Network_Init
layers = CNN_Network_Init(imageSize,NumofCategories,filterSize,numFilters);

% distributed random numbers with standard deviation of 0.0001 for first conv layer 
layers(2).Weights = 0.0001 * randn([filterSize depth numFilters]);

% Set the network training options
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 250, ...
    'MiniBatchSize', 128, ...
    'Verbose', true); 


 
if doTraining    
    % Train a network.
    cifar10Net = trainNetwork(TrainingImages, TrainingLabels, layers, opts);
    save cifar10Net.mat cifar10Net
else
    % Load pre-trained detector for the example.
    load('cifar10Net.mat','cifar10Net')      
end



%% Validate CIFAR-10 Network Training


% Extract the first convolutional layer weights
w = cifar10Net.Layers(2).Weights;
 
% rescale and resize the weights for better visualization
w = mat2gray(w);
w = imresize(w, [100 100]);
 
figure
montage(w)
 
% Run the network on the test set.
YTest = classify(cifar10Net, ValidationImages);
 
% Calculate the accuracy.
accuracy = sum(YTest == ValidationLabels)/numel(ValidationLabels)
