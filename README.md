# CIFAR-10 Submission

# Purpose

Created for the CIFAR-10 classification competition within the UT-MLDS student organization (Fall 2017 - Spring 2018).

# Brief Description

Architecture uses 18 layers based on [residual networks](https://arxiv.org/pdf/1512.03385.pdf). Specific architectural changes included the use of LeakyReLU as the activation function and employing a bilinear output layer. Grouped convolution was added in an effort to stabilize the bilinear layer (which was initially prone to exploding gradients). Training strategy involved occasionally varying data augmentation and weight decay between epochs.

*Note:* Grouped convolutions were used in the [ResNeXt](https://arxiv.org/pdf/1611.05431.pdf) architecture which is derivative of residual networks. Overall, the layout is more closely related to the original residual network paper.

# Score

![](https://github.com/jacoblubecki/cifar10-utmlds-s18/blob/master/score.png)
