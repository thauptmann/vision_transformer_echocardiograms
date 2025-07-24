# Vision Transformer for Echocardiogram Predictions
The purporse of this repository is to try different *Vision Transformer* (ViT) models on medical video data and to deepen my understanding of PyTorch and Lightning. The EchoNet-Dynamic[<sup>1</sup>](#References) dataset is available after registration and contains 10,030 apical-4-chamber echocardiography videos with corresponding *ejection fraction* (EF).

## Results
Mean squared error for Ejection Fraction prediction.

|Model         | MAE|
|:------        |:-------:|
|EchoNetDynamic| 4.1   |

## References
[1] D. Ouyang et al., ‘EchoNet-Dynamic: a Large New Cardiac Motion Video Data Resource for Medical Machine Learning’.
