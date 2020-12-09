#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/12/3 18:32
#@Author  :Wenbo 
#@FileName: test.py.py

import paddle

from paddle.metric import Accuracy
from paddle.vision.models import mobilenet_v2


import warnings
warnings.filterwarnings("ignore")


from dataset import MyDataset
import paddle.vision.transforms as T



transform = T.Compose([
    T.RandomResizedCrop([448, 448]),
    T.RandomHorizontalFlip(),
    T.RandomRotation(90),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),

])

train_dataset = MyDataset(txt=r'C:\Users\11982\Desktop\paddle\train.txt', transform=transform)

train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=8, shuffle=True, num_workers=8)

# build model
model = mobilenet_v2(pretrained=True,scale=1.0, num_classes=2, with_pool=True)

# 自定义Callback 记录训练过程中的loss信息
# class LossCallback(paddle.callbacks.Callback):
#
#     def on_train_begin(self, logs={}):
#         # 在fit前 初始化losses，用于保存每个batch的loss结果
#         self.losses = []
#
#         self.acc = []
#
#     def on_train_batch_end(self, step, logs={}):
#         # 每个batch训练完成后调用，把当前loss添加到losses中
#
#         self.losses.append(logs.get('loss'))
#         self.dataframe = pd.DataFrame(self.losses)
#         self.dataframe.to_csv('loss.csv')
#
#         self.acc.append(logs.get('acc_top1'))
#         self.dataframe1 = pd.DataFrame(self.acc)
#         self.dataframe1.to_csv('acc.csv')
# # 初始化一个loss_log 的实例，然后将其作为参数传递给fit
# loss_log = LossCallback()

# 调用飞桨框架的VisualDL模块，保存信息到目录中。
callback = paddle.callbacks.VisualDL(log_dir='visualdl_log_dir')

model = paddle.Model(model)
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())


# model.prepare()
# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy(topk=(1, 2))
    )

model.fit(train_loader,
        epochs=500,
        verbose=1,
        callbacks=callback
        )

# print(loss_log.losses)

model.evaluate(train_dataset, batch_size=8, verbose=1)

# model.save()

model.save('inference_model', False)



