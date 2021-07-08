import itertools
import os
import sys
import time
import datetime

def get_real_arch(arch, stages=[2, 3, 3]):
  arch = list(arch)
  result = ''
  for stage in stages:
    id_num = 0
    for idx in range(stage):
      op = arch.pop(0)
      if idx == 0:
        result += op
        continue
      if op != '0':
        result += op
      else:
        id_num += 1
    result += '0' * id_num
  return result

part = sys.argv[1]
start_idx = int(sys.argv[2])
num = int(sys.argv[3])
job_num = int(sys.argv[4])
save_path = sys.argv[5]
seed = sys.argv[6]
model = sys.argv[7]
choices = ['1', '2', '3', '4']
layers = 7
evaluating_archs = {}
Free_time_list = ['Sat', 'Sun']

for idx, arch in enumerate(itertools.product(*[choices]*layers)):
    now = datetime.datetime.now()
    if now.strftime('%a') in Free_time_list:
        job_num = 96
    else:
        if now.strftime('%H') == '21':
            job_num = 96
        elif now.strftime('%H') == '10':
            job_num = 96
    arch = ''.join(arch)
    if arch in evaluating_archs:
        print('Already evaluating.')
        continue
    evaluating_archs[arch] = 1
    if idx < start_idx:
        continue
    if idx == start_idx + num:
        break
    if os.path.exists(os.path.join(save_path, '{}.txt'.format(arch))):
        print('Already evaluated ({}/{}): {}'.format(idx, start_idx+num, arch))
        continue
    while int(os.popen('squeue | grep {} | grep suxiu | wc -l'.format(part)).read()) >= job_num:
        time.sleep(5)
    print('Evaluating ({}/{}): {}'.format(idx, start_idx+num, arch))
    os.system('nohup sh run.sh {} {} python train.py --arch {} --save {} --seed {} --model {}&'.format(part, arch, arch, save_path, seed, model))
    time.sleep(3)

print('Evaluate done. start_idx: {}, num: {}'.format(start_idx, num))


'''
for idx, arch in enumerate(itertools.product(*[choices]*layers)):
    arch = ''.join(arch)
    if arch in evaluating_archs:
        print('Already evaluating.')
        continue
    evaluating_archs[get_real_arch(arch)] = 1
    if idx < start_idx:
        continue
    if idx == start_idx + num:
        break
    if os.path.exists(os.path.join(save_path, '{}.txt'.format(get_real_arch(arch)))):
        print('Already evaluated.')
        continue
    while int(os.popen('squeue | grep {} | grep huangtao | wc -l'.format(part)).read()) >= job_num:
        time.sleep(5)
    print('Evaluating ({}/{}): {}'.format(idx, start_idx+num, arch))
    os.system('nohup sh run_name.sh {} {} python train.py --arch {} --save {} > /dev/zero &'.format(part, get_real_arch(arch), arch, save_path))
    time.sleep(3)

print('Evaluate done. start_idx: {}, num: {}'.format(start_idx, num))
'''

#python start_job.py VA 0 7000 64 bench-cifar10 srun