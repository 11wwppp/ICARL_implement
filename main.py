from iCaRL import iCaRLmodel
from ResNet import resnet18_cbam
import torch





numclass=10
feature_extractor=resnet18_cbam()
img_size=32
batch_size=128
task_size=10
memory_size=2000
epochs=100
learning_rate=2.0
'''
numclass:目前学习类别数
feature_extractor:特征提取网络
batch_size:批大小
memory_size:记忆库大小
task_size:每次增量学习新增类别数
'''
model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)#这是
#model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))

for i in range(10): # 循环10次，每次增加task_size个新类别
    print('before train the %d time'%(i+1))
    model.beforeTrain()
    print('train the %d time'%(i+1))
    accuracy=model.train()
    print('after train the %d time'%(i+1))
    model.afterTrain(accuracy)
