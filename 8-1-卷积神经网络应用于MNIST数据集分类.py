# coding: utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import Adam

print(tf.version.VERSION)

mnist = tf.keras.datasets.mnist                            # load MNSIT
(x_train, y_train), (x_test, y_test) = mnist.load_data()   # Load train dataset & test dataset 载入数据，数据载入的时候就已经划分好训练集和测试集

print(x_train.shape, y_train.shape)                        # 输出训练集样本和标签的大小
print(x_train[0])                                          # 查看数据，例如训练集中第一个样本的内容和标签
print(y_train[0])

# 可视化样本，下面是输出了训练集中前4个样本
fig, ax = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
ax = ax.flatten()
for i in range(4):
    img = x_train[i].reshape(28, 28)
    # ax[i].imshow(img,cmap='Greys')
    ax[i].imshow(img)
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


# 把数据reshape变成4维数据(数据数量，图片高度，图片宽度，图片通道数)
x_train = x_train.reshape(-1,28,28,1)/255.0               # 这里要注意，在tensorflow中，在做卷积的时候需要把数据变成4维的格式，黑白图片的通道数是1，彩色图片通道数是3
x_test  = x_test.reshape(-1,28,28,1)/255.0                # MNIST中的图像默认为uint8（0-255的数字）。/255.0 将其归一化到0-1之间的浮点数

# 把训练集和测试集的标签转为独热编码
y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=10)

print(x_train.shape, y_train.shape)                        # 输出训练集样本和标签的大小
print(x_train[0])                                          # 查看数据，例如训练集中第一个样本的内容和标签
print(y_train[0])

# 定义顺序模型
model = Sequential()

ckpt = tf.train.Checkpoint(model=model)

model.add(Convolution2D( input_shape = (28,28,1), filters = 32, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu')) # 第一个卷积层： input_shape 输入数据；filters 滤波器个数32，生成32张特征图 ；kernel_size 卷积窗口大小5*5；strides 步长1；padding padding方式 same/valid；activation 激活函数
model.add(MaxPooling2D( pool_size = 2,strides = 2,padding = 'same',))       # 第一个池化层：pool_size 池化窗口大小2*2； strides 步长2；padding padding方式 same/valid
model.add(Convolution2D(64,5,strides=1,padding='same',activation='relu'))   # 第二个卷积层 ： filters 滤波器个数64，生成64张特征图 ；kernel_size 卷积窗口大小5*5 ；strides 步长1 ；padding padding方式 same/valid ；activation 激活函数
model.add(MaxPooling2D(2,2,'same'))                                         # 第二个池化层 ： pool_size 池化窗口大小2*2 ；strides 步长2 ；padding padding方式 same/valid
model.add(Flatten())                                                        # 把第二个池化层的输出进行数据扁平化，相当于把(64,7,7,64)数据->(64,7*7*64)
model.add(Dense(1024,activation = 'relu'))                                  # 第一个全连接层
model.add(Dropout(0.5))                                                     # Dropout
model.add(Dense(10,activation='softmax'))                                   # 第二个全连接层

adam = Adam(lr=1e-4)                                                                    # 定义优化器
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])      # 定义优化器，loss function，训练过程中计算准确率

model.summary()                                                                         # Display the model's architecture


# manager = tf.train.CheckpointManager(ckpt, directory='./CheckPointTest', max_to_keep=5)


model.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=(x_test, y_test))     # 训练模型

# model.save('mnist.h5')                                                                  # 保存模型
# manager.save(checkpoint_number=100)
ckpt.save('./CheckPointTest/model123456.ckpt')