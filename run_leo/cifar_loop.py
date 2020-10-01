# Running this script in sbatch will train multiple neural networks on the same gpu.

import datetime

import subprocess
subprocess.call('pwd')

print('running cifar pooling sweep in parallel')

jobs = []
lr = 0.1
momentum = 0.2

time_str = str(datetime.datetime.today())
with open("p_" + 'mean' + time_str + ".txt", "w") as f:
    jobs.append(subprocess.Popen(['python', '../train_cifar.py', 
                                  '--pooling_type', 'max',
                                  '--lr', str(lr),
                                  '--momentum', str(momentum),
                                  '--tensorboard'], stdout=f))
with open("p_" + 'avg' + time_str + ".txt", "w") as f:
    jobs.append(subprocess.Popen(['python', '../train_cifar.py',
                                  '--pooling_type', 'avg',
                                  '--lr', str(lr),
                                  '--momentum', str(momentum),
                                  '--tensorboard'], stdout=f))
with open("p_" + 'wavelet' + time_str + ".txt", "w") as f:
    jobs.append(subprocess.Popen(['python', '../train_cifar.py',
                                  '--pooling_type', 'wavelet',
                                  '--lr', str(lr),
                                  '--momentum', str(momentum),
                                  '--tensorboard'], stdout=f))
with open("p_" + 'adaptive_wavelet' + time_str + ".txt", "w") as f:
    jobs.append(subprocess.Popen(['python', '../train_cifar.py',
                                  '--lr', str(lr),
                                  '--momentum', str(momentum),
                                  '--pooling_type', 'adaptive_wavelet',
                                  '--tensorboard'], stdout=f))
with open("p_" + 'scaled_wavelet' + time_str + ".txt", "w") as f:
    jobs.append(subprocess.Popen(['python', '../train_cifar.py',
                                  '--lr', str(lr),
                                  '--momentum', str(momentum),
                                  '--pooling_type', 'scaled_wavelet',
                                  '--tensorboard'], stdout=f))

for job in jobs:
    job.wait()