# Running this script in sbatch will train multiple neural networks on the same gpu.

import datetime

import subprocess
subprocess.call('pwd')

print('running cifar pooling sweep in parallel')

jobs = []

time_str = str(datetime.datetime.today())
with open("b_" + experiment + time_str + ".txt", "w") as f:
    jobs.append(subprocess.Popen(['python', '../train_cifar.py', 'pooling_type', 'mean'], stdout=f))
    jobs.append(subprocess.Popen(['python', '../train_cifar.py', 'pooling_type', 'avg'], stdout=f))
    jobs.append(subprocess.Popen(['python', '../train_cifar.py', 'pooling_type', 'wavelet'], stdout=f))
    jobs.append(subprocess.Popen(['python', '../train_cifar.py', 'pooling_type', 'adaptive_wavelet'], stdout=f))
    jobs.append(subprocess.Popen(['python', '../train_cifar.py', 'pooling_type', 'scaled_wavelet'], stdout=f))

for job in jobs:
    job.wait()
