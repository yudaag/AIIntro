# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 기존 학습된 파라미터 로드

sys.path.insert(0, project_root)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet_my import DeepConvNet
from common.trainer import Trainer
np.random.seed(0)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet() 
network.load_params("cnn_params.pkl")
print("Loaded pretrained parameters!")


trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=256,
                  optimizer='adamw',
                    optimizer_param={
                        'lr': 1e-3,
                        'weight_decay': 1e-4
                    },
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 정확도 보관
# np.save("train_acc.npy", trainer.train_acc_list)
# np.save("test_acc.npy", trainer.test_acc_list)

# 매개변수 보관
network.save_params("cnn_params.pkl")
print("Saved Network Parameters!")


# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(trainer.train_acc_list))

plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)

plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title("Fine-tuning Accuracy")

plt.show()