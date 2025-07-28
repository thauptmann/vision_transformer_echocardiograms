# Vision Transformer for Ejection Fraction Prediction from Echocardiograms
The purporse of this repository is to try different *Vision Transformer* (ViT) models on medical video data and to deepen my understanding of PyTorch and Lightning. The EchoNet-Dynamic [[1]](#References) dataset is available after registration and contains 10,030 apical-4-chamber echocardiography videos with corresponding *ejection fraction* (EF).

The original model was a two-step method, where first segmentations of the left heart chamber were computed to extract individual heartbeats and used as input gor the convolutional neural network. The final result was the mean over several subvideos. I omit this step and directly feed the whole echocardiogram video into the neural network.

## Results
Mean squared error for Ejection Fraction prediction.

|Model         | MAE|
|:------        |:-------:|
|EchoNet-Dynamic [[2]](#References)| 4.1[^1]|

[^1]: Taken from publication.

## References
[1] D. Ouyang et al., ‘EchoNet-Dynamic: a Large New Cardiac Motion Video Data Resource for Medical Machine Learning’.

[2] D. Ouyang et al., ‘Video-based AI for beat-to-beat assessment of cardiac function’, Nature, vol. 580, no. 7802, pp. 252–256, Apr. 2020, doi: 10.1038/s41586-020-2145-8.
