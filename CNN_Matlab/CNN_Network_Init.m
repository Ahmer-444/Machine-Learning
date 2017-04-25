function Layers = CNN_Network_Init(InputSize,NumofCategories,filterSize,numFilters)

    % Create the input data layer that will feed to our CNN Network. Dimesnions = 32 x 32 x 3
    inputLayer = imageInputLayer(InputSize);

    % Create hidden/middle layers of CNN network that will be consist of repetition of Convolutional, 
    % ReLU (activation function)and Pooling Layers


    middleLayers = [

    convolution2dLayer(filterSize, numFilters, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(3, 'Stride', 2)
    
    % Repeat the 3 core layers to complete the middle of the network.
    convolution2dLayer(filterSize, numFilters, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(3, 'Stride',2)

    convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(3, 'Stride',2)

    ]

    % The final layers of a CNN are typically composed of fully connected
    % layers and a softmax loss layer.

    finalLayers = [

    % Add a fully connected layer with 64 output neurons. The output size of
    % this layer will be an array with a length of 64.
    fullyConnectedLayer(64)

    % Add an ReLU non-linearity.
    reluLayer

    % Add the last fully connected layer. At this point, the network must
    % produce 10 signals that can be used to measure whether the input image
    % belongs to one category or another. This measurement is made using the
    % subsequent loss layers.
    fullyConnectedLayer(NumofCategories)

    % Add the softmax loss layer and classification layer. The final layers use
    % the output of the fully connected layer to compute the categorical
    % probability distribution over the image classes. During the training
    % process, all the network weights are tuned to minimize the loss over this
    % categorical distribution.
    softmaxLayer
    classificationLayer
    ]

    % Combine the input, middle, and final layers.
    Layers = [
        inputLayer
        middleLayers
        finalLayers
        ]
end