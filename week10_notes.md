##生成tfrecord
使用 convert_fcn_dataset.py脚本生成tfrecord

###其中会报没有cv2这个模块
使用以下命令安装opencv 
```
pip3 install opencv-python
```
**生成的tfrecord如下所示**
![](/home/leemax/图片/tfrecord.png) 


##搭建FCN 8s网络
####搭建过程中使用的代码
**tf.nn.conv2d_transpose**反卷积，用来进行上采样
```
upsampled_logits = tf.nn.conv2d_transpose(logits, 
					upsample_filter_tensor_x2,
					output_shape=tf.shape(aux_logits_16s),
					strides=[1, 2, 2, 1],
					padding='SAME')
```
>logits:作为转置卷积的**input**
>upsample_filter_tensor_x2：转置卷积的**kernel**，此处为双线性插值
>**output_shape**:指定的大小，会根据大小对相应位置进行裁切
>**stride**:四维张量，前后都为1，中间两位分别为水平和垂直步长
>**padding**

####具体实现过程
1.修改最后上采样的倍数
2.将16s最后一层16×上采样之前的logits，作为8s输入的一部分，对其进行2×上采样
3.将2×上采样后的logits与pool3后的全卷积做element_wise相加。
4.将上述相加后的结果进行8×上采样作为最后的logits
5.送入模型
#####修改的代码如修改过的train.py文件中所示
##训练过程
###下载ckpt

使用如下命令进行训练：
```
