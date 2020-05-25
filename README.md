# SR
超分辨率技术（Super-Resolution, SR）

**论文 [Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092)的keras复现**

注：因为自己的windows和实验室的台式机都在学校，同时实验室的服务器由于很多学长学姐在使用，远程账号连接不稳定，所以刚开始是在自己的mac上用keras实现，CPU运行。

（要求：注意平均不能比他差0.01db）

#### 结果：                

| Set-5 scale=3 | bicubic | Paper-SRCNN | self-bicubic | self-SRCNN |
| ------------- | ------- | ----------- | ------------ | ---------- |
| baby          | 33.91   | 35.01       | 32.45        |            |
| birds         | 32.58   | 34.91       | 31.34        |            |
| butterfly     | 24.04   | 27.58       | 22.95        |            |
| head          | 32.88   | 33.55       | 31.46        |            |
| Woman         | 28.56   | 30.92       | 26.93        |            |
| average       | 30.39   | 32.39       | 29.026       |            |

#### 实现过程的问题：

1.loss下降的不快，但是准确率一直稳定在0.1，检查了train和label确实是对应上的，batchsize调小影响也不大，去掉了输入图片/255的归一化，归一化之后，loss太小了，loss已经1e-3, acc才1e-4；虽然准确率一直在0.1抖动，但是简单的查看了一下模型效果也还行，butterfly的PSNR为23.7；

2.预测过程：图片YCrCb空间的问题，图像还原不了RGB；Y是指亮度(luma)分量(灰阶值)，Cb指蓝色色度分量，而Cr指红色色度分量；训练只用了Y，当然还原不出来，刚开始没弄清楚这个YCrCb，要想RGB图片，只需要调整一下网络的channel=3，同时对应调整一下train和label就行了；

3.acc还是一直上不去，因为之前我一直训练的epoch比较短，epoch=6左右，简单的看一下loss和acc的情况，因为网络非常简单，需要拟合的参数很少，我觉得几个epoch应该就够了；实在不知道网络的问题在哪，应该是数据量比较大21884，几个epoch网络学不到数据的特征，所以移到了服务器上，跑了200个epoch，结果还是一样的问题，acc在0.1左右抖动；

4.PSNR的计算结果有问题，可能的原因：因为测试的时候，是拼接来完成的，所以图像的周围像素点是直接去掉了，butterfly的bicubic的PSNR:22.95；PSNR的计算函数有问题(排除，tf中可以直接调用)；

5.从头到尾仔细检查了几遍代码和模型，突然发现可能是keras评估函数的acc的问题，想取200epoch中最好的model试一下PSNR，结果服务器连不上了；直接把**训练过程的评估函数metrics从acc换成了PSNR**，终于解决了，PSNR一直上升到30左右，和论文差不多，但是有待提升，等候服务器连接正常；