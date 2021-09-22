# coding: utf-8
import pprint
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import Adam


# MNIST_data指的是存放数据的文件夹路径，one_hot=True 为采用one_hot的编码方式编码标签
# mnist = input_data.read_data_sets('../datasets/MNIST_data/', one_hot=True)
mnist = tf.keras.datasets.mnist
# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)   # 输出训练集样本和标签的大小
# 查看数据，例如训练集中第一个样本的内容和标签
print(x_train[0])       # 是一个包含784个元素且值在[0,1]之间的向量
print(y_train[0])

# pprint.pprint(x_train[0],width=300)       # 是一个包含784个元素且值在[0,1]之间的向量
# pprint.pprint(y_train[0],width=300)

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


# mnist = tf.keras.datasets.mnist                                                 # 载入数据
# (x_train, y_train), (x_test, y_test) = mnist.load_data()                        # 载入数据，数据载入的时候就已经划分好训练集和测试集
#
# # 把数据reshape变成4维数据
# x_train = x_train.reshape(-1,28,28,1)/255.0                                     # 这里要注意，在tensorflow中，在做卷积的时候需要把数据变成4维的格式(数据数量，图片高度，图片宽度，图片通道数)，黑白图片的通道数是1，彩色图片通道数是3
# x_test  = x_test.reshape(-1,28,28,1)/255.0
#
# # 把训练集和测试集的标签转为独热编码
# y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)
# y_test = tf.keras.utils.to_categorical(y_test,num_classes=10)
#
# # 定义顺序模型
# model = Sequential()
#
# # 第一个卷积层： input_shape 输入数据；filters 滤波器个数32，生成32张特征图 ；kernel_size 卷积窗口大小5*5；strides 步长1；padding padding方式 same/valid；activation 激活函数
# model.add(Convolution2D( input_shape = (28,28,1), filters = 32, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
# # 第一个池化层：pool_size 池化窗口大小2*2； strides 步长2；padding padding方式 same/valid
# model.add(MaxPooling2D( pool_size = 2,strides = 2,padding = 'same',))
#
# # 第二个卷积层 ： filters 滤波器个数64，生成64张特征图 ；kernel_size 卷积窗口大小5*5 ；strides 步长1 ；padding padding方式 same/valid ；activation 激活函数
# model.add(Convolution2D(64,5,strides=1,padding='same',activation='relu'))
# # 第二个池化层 ： pool_size 池化窗口大小2*2 ；strides 步长2 ；padding padding方式 same/valid
# model.add(MaxPooling2D(2,2,'same'))
# model.add(Flatten())                                                                    # 把第二个池化层的输出进行数据扁平化，相当于把(64,7,7,64)数据->(64,7*7*64)
#
# model.add(Dense(1024,activation = 'relu'))                                              # 第一个全连接层
# model.add(Dropout(0.5))                                                                 # Dropout
# model.add(Dense(10,activation='softmax'))                                               # 第二个全连接层
#
# adam = Adam(lr=1e-4)                                                                    # 定义优化器
# model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])      # 定义优化器，loss function，训练过程中计算准确率
# model.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=(x_test, y_test))     # 训练模型
#
# model.save('mnist.h5')                                                                  # 保存模型

