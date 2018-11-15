## 生成tfrecord
使用 convert_fcn_dataset.py脚本生成tfrecord

### 其中会报没有cv2这个模块
使用以下命令安装opencv 
```
pip3 install opencv-python
```
**生成的tfrecord如下所示**
 ![](https://i.imgur.com/9uTajqU.png)


## 搭建FCN 8s网络
#### 搭建过程中使用的代码
**tf.nn.conv2d_transpose**反卷积，用来进行上采样
```
upsampled_logits = tf.nn.conv2d_transpose(logits, 
					upsample_filter_tensor_x2,
					output_shape=tf.shape(aux_logits_16s),
					strides=[1, 2, 2, 1],
					padding='SAME')
```

>logits:作为转置卷积的**input**
>
>upsample_filter_tensor_x2：转置卷积的**kernel**，此处为双线性插值
>
>**output_shape**:指定的大小，会根据大小对相应位置进行裁切
>
>**stride**:四维张量，前后都为1，中间两位分别为水平和垂直步长
>
>**padding**

#### 具体实现过程
1.修改最后上采样的倍数

2.将16s最后一层16×上采样之前的logits，作为8s输入的一部分，对其进行2×上采样

3.将2×上采样后的logits与pool3后的全卷积做element_wise相加。

4.将上述相加后的结果进行8×上采样作为最后的logits

5.送入模型
##### 修改的代码如修改过的train.py文件中所示
## 训练过程
### 下载ckpt
**使用modelzoo下载ckpt**
在[TensorFlow的slim](https://github.com/tensorflow/models/tree/master/research/slim)框架下的modelzoo下载相应的ckpt
### 训练
由于没带自己电脑，选择使用tinymind进行训练过程。按照作业要求搭建模型后，上传自己代码进行训练，如下为训练过程[log](https://www.tinymind.com/executions/jgwjpcdp)：
![](https://i.imgur.com/wzaU0tG.png)
但是在tinymind上输出只有train没有eval：
![](https://i.imgur.com/uVi4bD3.png)
但是代码中确实有eval的输出，所以不知道是为何。。。
![](https://i.imgur.com/vneWIBk.png)


### 心得体会
对于8xfcn的原理，大概了解是为了得到更为稠密的featuremap，但是为什么采用element_wise相加的原理并不是特别清楚，但是想了一下好像除了相加之外也没有任何其他更好的办法。但个人觉得是不是相加之后最后的值还要根据相加的层数进行一定的标准化？比如限制范围到0-255之间，个人理解上是这样，但是找了一圈代码好像并没有找到相关实现，还需继续思考和查看资料。

最后不知为何没有eval的输出，然后也没有报任何错误，目测可能跟tinymind的配置有关系。准备回去本地跑一下看看，如有结果再进行更新。
