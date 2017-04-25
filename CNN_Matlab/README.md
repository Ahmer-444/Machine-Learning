Its the implementation of CNN in Matlab for CIFAR-10 subset. I haven't focus on the accuracy but on the structured implementation. We can increase the accuracy by increasing the "Epchos" and not a big deal
You can check the visualization for data and weights in "TrainigData(1)+WeightsVisualization.jpg".
Some of the network architecture information and training output are as follows: 

layers = 

  15x1 Layer array with layers:

     1   ''   Image Input             32x32x3 images with 'zerocenter' normalization
     2   ''   Convolution             32 5x5 convolutions with stride [1  1] and padding [2  2]
     3   ''   ReLU                    ReLU
     4   ''   Max Pooling             3x3 max pooling with stride [2  2] and padding [0  0]
     5   ''   Convolution             32 5x5 convolutions with stride [1  1] and padding [2  2]
     6   ''   ReLU                    ReLU
     7   ''   Max Pooling             3x3 max pooling with stride [2  2] and padding [0  0]
     8   ''   Convolution             64 5x5 convolutions with stride [1  1] and padding [2  2]
     9   ''   ReLU                    ReLU
    10   ''   Max Pooling             3x3 max pooling with stride [2  2] and padding [0  0]
    11   ''   Fully Connected         64 fully connected layer
    12   ''   ReLU                    ReLU
    13   ''   Fully Connected         10 fully connected layer
    14   ''   Softmax                 softmax
    15   ''   Classification Output   cross-entropy
|=========================================================================================|
|     Epoch    |   Iteration  | Time Elapsed |  Mini-batch  |  Mini-batch  | Base Learning|
|              |              |  (seconds)   |     Loss     |   Accuracy   |     Rate     |
|=========================================================================================|
|            1 |           50 |         6.06 |       2.3027 |       10.16% |     0.001000 |
|            2 |          100 |        12.18 |       2.3026 |       11.72% |     0.001000 |
|            2 |          150 |        18.29 |       2.3027 |        9.38% |     0.001000 |
|            3 |          200 |        24.41 |       2.3027 |        8.59% |     0.001000 |
|            4 |          250 |        30.53 |       2.3026 |        7.03% |     0.001000 |
|            4 |          300 |        36.64 |       2.3027 |        9.38% |     0.001000 |
|            5 |          350 |        42.76 |       2.3025 |       10.94% |     0.001000 |
|            6 |          400 |        48.88 |       2.3026 |       11.72% |     0.001000 |
|            6 |          450 |        54.99 |       2.3026 |        8.59% |     0.001000 |
|            7 |          500 |        61.11 |       2.3025 |       11.72% |     0.001000 |
|            8 |          550 |        67.25 |       2.3026 |        7.03% |     0.001000 |
|            8 |          600 |        73.42 |       2.3026 |        8.59% |     0.001000 |
|            9 |          650 |        79.62 |       2.3026 |        5.47% |     0.000100 |
|            9 |          700 |        85.71 |       2.3026 |        7.81% |     0.000100 |
|           10 |          750 |        91.86 |       2.3026 |        3.91% |     0.000100 |


# Validation Accuracy
accuracy =

    0.78
