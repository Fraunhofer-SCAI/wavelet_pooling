
import datetime

import subprocess
subprocess.call('pwd')

print('running mnist parameter sweep in parallel')


lr_lst = ['0.12', '0.1', '0.08', '0.05']
gamma_lst = ['1.0', '0.98', '0.95']
momentum_lst = ['0.6']

for lr in lr_lst:
    for gamma in gamma_lst:
        for momentum in momentum_lst:
            jobs = []
            time_str = str(datetime.datetime.today())
            print('lr', lr, 'g', gamma, 'm', momentum, ' at time:', time_str)
            with open("log/avg_" + "lr_" + str(lr) + '_gamma_' + str(gamma) 
                    + '_' + time_str + ".txt", "w") as f:
                jobs.append(subprocess.Popen(['python', '../mnist_pool.py', '--lr', lr,
                                            '--gamma', gamma, '--momentum', momentum, 
                                            '--pooling_type', 'adaptive_wavelet',
                                            '--tensorboard'], stdout=f))
            with open("log/scaled_wavelet_" + "lr_" + str(lr) + '_gamma_' + str(gamma) 
                    + '_' + time_str + ".txt", "w") as f:
                jobs.append(subprocess.Popen(['python', '../mnist_pool.py', '--lr', lr,
                                            '--gamma', gamma,  '--momentum', momentum, 
                                            '--pooling_type', 'scaled_adaptive_wavelet',
                                            '--tensorboard'], stdout=f))
            with open("log/wavelet_" + "lr_" + str(lr) + '_gamma_' + str(gamma)
                    + '_' + time_str + ".txt", "w") as f:
                jobs.append(subprocess.Popen(['python', '../mnist_pool.py', '--lr', lr,
                                            '--gamma', gamma,  '--momentum', momentum, 
                                            '--pooling_type', 'avg',
                                            '--tensorboard'], stdout=f))
            with open("log/wavelet_" + "lr_" + str(lr) + '_gamma_' + str(gamma)
                    + '_' + time_str + ".txt", "w") as f:
                jobs.append(subprocess.Popen(['python', '../mnist_pool.py', '--lr', lr,
                                            '--gamma', gamma,  '--momentum', momentum, 
                                            '--pooling_type', 'max',
                                            '--tensorboard'], stdout=f))

            for job in jobs:
                job.wait()
