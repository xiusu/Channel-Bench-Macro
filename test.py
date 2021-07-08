import json
import itertools
from numpy import *
import os


# choices = ['1', '2', '3', '4']
# layers = 7
# pre_locate1 = 'bench-resnet_1/'
# pre_locate2 = 'bench-resnet_2/'
# pre_locate3 = 'bench-resnet_3/'

# final_json_mobilenet = {}

# for idx, arch in enumerate(itertools.product(*[choices]*layers)):
#     if idx % 100 == 0:
#         print('idx:{}'.format(idx))
#     arch = ''.join(arch)
#     now1 = pre_locate1 + arch + '.txt'
#     now2 = pre_locate2 + arch + '.txt'
#     now3 = pre_locate3 + arch + '.txt'
#     f1 = open(now1)
#     t1 = json.load(f1)
    
#     f2 = open(now2)
#     t2 = json.load(f2)

#     f3 = open(now3)
#     t3 = json.load(f3)

#     t1['test_acc'] = t1['test_acc'] + t2['test_acc'] + t3['test_acc']
#     t1['mean'] = mean(t1['test_acc'])
#     t1['std'] = std(t1['test_acc'], ddof=1)

#     final_json_mobilenet[t1['arch']] = {}
#     final_json_mobilenet[t1['arch']]['acc'] = t1['test_acc']
#     final_json_mobilenet[t1['arch']]['flops'] = t1['flops']
#     final_json_mobilenet[t1['arch']]['params'] = t1['params']
#     final_json_mobilenet[t1['arch']]['mean'] = t1['mean']
#     final_json_mobilenet[t1['arch']]['std'] = t1['std']

# with open("Resuls_ResNet.json", "w") as f:
#     json.dump(final_json_mobilenet, f)

# print("finish")


a = open("bench-cifar10_1/log/1111111.log")
b = a.readlines()
print(b[0])