# 数据集是来自Visual Geometry Group 的 17 Category Flower Dataset据集，也就是 17 种花的数据集。具体是哪 17 种这个我们可以不用管，反正就是 17 个类别。
# 每个类别的花有 80 张图片，一共是 1360 张图片。观察图片名称我们可以发现是都是由编号构成，前 1-80 号为第一种花，81 到 160 号为第二种花以此类推，1360 张图片一共 17 种花。
# 下载如下URL中的数据集后，解压到./17_Category_Flower_Dataset文件夹中
# URL:https://www.robots.ox.ac.uk/~vgg/data/flowers/17/

import os
import shutil
import random
import numpy as np


DATASET_DIR = '17_Category_Flower_Dataset'                  # 数据集路径
MODEL_DATASET_DIR = DATASET_DIR+"/new_17_flowers"           # 数据集路径
NEW_DIR = DATASET_DIR+"/data"                               # 数据切分后存放路径
num_test = 0.2                                              # 测试集占比

# 根据文件名称将数据集分为17个类别
def Classification_by_filename():
    if os.path.exists(DATASET_DIR+'/new_17_flowers'):
        print('new_17_flowers exit! deleting...')
        shutil.rmtree(DATASET_DIR+'/new_17_flowers')
    os.mkdir(DATASET_DIR+'/new_17_flowers')                                                                         # 新建文件夹用于存放整理后的图片

    for i in range(17): os.mkdir(DATASET_DIR+'/new_17_flowers' + '/' + 'flower' + str(i))                           # 17个种类新建17个文件夹0-16

    file_name = os.listdir(DATASET_DIR+'/jpg/')
    file_name.sort()  # 对文件名排序

    for i, path in enumerate(file_name):                                                                            # 循环所有花的图片
        image_path = DATASET_DIR+'/jpg/' + path                                                                     # 定义花的图片完整路径
        shutil.copyfile(image_path, DATASET_DIR+'/new_17_flowers' + '/' + 'flower' + str(i // 80) + '/' + path)     # 复制到对应类别，每个类别80张图片

# 打乱所有种类数据，并分割训练集和测试集
def shuffle_all_files(dataset_dir, new_dir, num_test):
    # 先删除已有new_dir文件夹
    if not os.path.exists(new_dir):
        pass
    else:
        shutil.rmtree(new_dir)                              # 递归删除文件夹

    os.makedirs(new_dir)                                    # 重新创建new_dir文件夹
    train_dir = os.path.join(new_dir, 'train')              # 在new_dir文件夹目录下创建train文件夹
    os.makedirs(train_dir)
    test_dir = os.path.join(new_dir, 'test')                # 在new_dir文件夹目录下创建test文件夹
    os.makedirs(test_dir)

    directories = []                                        # 原始数据类别列表
    train_directories = []                                  # 新训练集类别列表
    test_directories = []                                   # 新测试集类别列表
    class_names = []                                        # 类别名称列表

    for filename in os.listdir(dataset_dir):                # 循环所有类别
        path = os.path.join(dataset_dir, filename)          # 原始数据类别路径
        train_path = os.path.join(train_dir, filename)      # 新训练集类别路径
        test_path = os.path.join(test_dir, filename)        # 新测试集类别路径

        if os.path.isdir(path):                             # 判断该路径是否为文件夹
            directories.append(path)                        # 加入原始数据类别列表
            train_directories.append(train_path)            # 加入新训练集类别列表
            os.makedirs(train_path)                         # 新建类别文件夹
            test_directories.append(test_path)              # 加入新测试集类别列表
            os.makedirs(test_path)                          # 新建类别文件夹
            class_names.append(filename)                    # 加入类别名称列表
    print('类别列表：', class_names)

    # 循环每个分类的文件夹
    for i in range(len(directories)):
        photo_filenames = []                                                    # 保存原始图片路径
        train_photo_filenames = []                                              # 保存新训练集图片路径
        test_photo_filenames = []                                               # 保存新测试集图片路径
        # 得到所有图片的路径
        for filename in os.listdir(directories[i]):
            path = os.path.join(directories[i], filename)                       # 原始图片路径
            train_path = os.path.join(train_directories[i], filename)           # 训练图片路径
            test_path = os.path.join(test_directories[i], filename)             # 测试集图片路径
            photo_filenames.append(path)                                        # 保存图片路径
            train_photo_filenames.append(train_path)
            test_photo_filenames.append(test_path)
        # list转array
        photo_filenames = np.array(photo_filenames)
        train_photo_filenames = np.array(train_photo_filenames)
        test_photo_filenames = np.array(test_photo_filenames)
        # 打乱索引
        index = [i for i in range(len(photo_filenames))]
        random.shuffle(index)
        # 对3个list进行相同的打乱，保证在3个list中索引一致
        photo_filenames = photo_filenames[index]
        train_photo_filenames = train_photo_filenames[index]
        test_photo_filenames = test_photo_filenames[index]

        test_sample_index = int((1 - num_test) * float(len(photo_filenames)))           # 计算测试集数据个数
        for j in range(test_sample_index, len(photo_filenames)):   shutil.copyfile(photo_filenames[j], test_photo_filenames[j])         # 复制测试集图片
        for j in range(0, test_sample_index):   shutil.copyfile(photo_filenames[j], train_photo_filenames[j])                           # 复制训练集图片

# 打乱并切分数据集
# Classification_by_filename()
# shuffle_all_files(MODEL_DATASET_DIR, NEW_DIR, num_test)

# ---------------------------------------------------↑↑↑ 以上代码为数据整理 ↑↑↑------------------------------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image   import ImageDataGenerator
from tensorflow.keras.utils                 import to_categorical
from tensorflow.keras.models                import Sequential
from tensorflow.keras.layers                import Dense,Dropout,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.optimizers            import Adam
from tensorflow.keras.callbacks             import LearningRateScheduler



num_classes = 17        # 类别数
batch_size = 32         # 批次大小
epochs = 100            # 周期数
image_size = 224        # 图片大小


# 训练集数据进行数据增强
train_datagen = ImageDataGenerator(
    rotation_range = 20,     # 随机旋转度数
    width_shift_range = 0.1, # 随机水平平移
    height_shift_range = 0.1,# 随机竖直平移
    rescale = 1/255,         # 数据归一化
    shear_range = 10,        # 随机错切变换
    zoom_range = 0.1,        # 随机放大
    horizontal_flip = True,  # 水平翻转
    brightness_range=(0.7, 1.3), # 亮度变化
    fill_mode = 'nearest',   # 填充方式
)
# 测试集数据只需要归一化就可以
test_datagen = ImageDataGenerator(
    rescale = 1/255,         # 数据归一化
)


# 训练集数据生成器，可以在训练时自动产生数据进行训练
# 从'data/train'获得训练集数据
# 获得数据后会把图片resize为image_size×image_size的大小

# generator每次会产生batch_size个数据
train_generator = train_datagen.flow_from_directory(
    '17_Category_Flower_Dataset/data/train',
    target_size=(image_size,image_size),
    batch_size=batch_size,
    )

# 测试集数据生成器
test_generator = test_datagen.flow_from_directory(
    '17_Category_Flower_Dataset/data/test',
    target_size=(image_size,image_size),
    batch_size=batch_size,
    )


# 字典的键为17个文件夹的名字，值为对应的分类编号
train_generator.class_indices


# AlexNet
model = Sequential()
# 卷积层
model.add(Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),padding='valid',input_shape=(image_size,image_size,3),activation='relu'))
model.add(MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid'))
model.add(Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid'))
model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2),padding='valid'))
# 全连接层
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# 模型概要
model.summary()


# In[7]:


# 学习率调节函数，逐渐减小学习率
def adjust_learning_rate(epoch):
    # 前30周期
    if epoch<=30:
        lr = 1e-4
    # 前30到70周期
    elif epoch>30 and epoch<=70:
        lr = 1e-5
    # 70到100周期
    else:
        lr = 1e-6
    return lr


# In[8]:


# 定义优化器
adam = Adam(lr=1e-4)

# 定义学习率衰减策略
callbacks = []
callbacks.append(LearningRateScheduler(adjust_learning_rate))

# 定义优化器，loss function，训练过程中计算准确率
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

# Tensorflow2.1版本之前可以使用fit_generator训练模型
# history = model.fit_generator(train_generator,steps_per_epoch=len(train_generator),epochs=epochs,validation_data=test_generator,validation_steps=len(test_generator))

# Tensorflow2.1版本(包括2.1)之后可以直接使用fit训练模型
history = model.fit(x=train_generator,epochs=epochs,validation_data=test_generator,callbacks=callbacks)


# In[9]:


# 画出训练集准确率曲线图
plt.plot(np.arange(epochs),history.history['accuracy'],c='b',label='train_accuracy')
# 画出验证集准确率曲线图
plt.plot(np.arange(epochs),history.history['val_accuracy'],c='y',label='val_accuracy')
# 图例
plt.legend()
# x坐标描述
plt.xlabel('epochs')
# y坐标描述
plt.ylabel('accuracy')
# 显示图像
plt.show()
# 模型保存
model.save('AlexNet.h5')