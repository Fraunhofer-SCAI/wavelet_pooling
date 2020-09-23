
import datetime

import subprocess
subprocess.call('pwd')

print('running mnist batch size sweep in parallel')


lr_lst = ['0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1']
gamma_lst = ['1.0', '0.99', '0.98', '0.97', '0.96', '0.95', '0.94', '0.93', '0.92', '0.91', '0.90']

for lr in lr_lst:
    for gamma in gamma_lst:
        jobs = []
        time_str = str(datetime.datetime.today())
        print(lr, gamma, ' at time:', time_str)
        with open("log/avg_" + "lr_" + str(lr) + '_gamma_' + str(gamma) 
                  + '_' + time_str + ".txt", "w") as f:
            jobs.append(subprocess.Popen(['python', '../mnist_pool.py', '--lr', lr,
                                          '--gamma', gamma, '--pooling_type', 'avg',
                                          '--tensorboard'], stdout=f))
        with open("log/scaled_wavelet_" + "lr_" + str(lr) + '_gamma_' + str(gamma) 
                  + '_' + time_str + ".txt", "w") as f:
            jobs.append(subprocess.Popen(['python', '../mnist_pool.py', '--lr', lr,
                                          '--gamma', gamma, '--pooling_type', 'scaled_wavelet',
                                          '--tensorboard'], stdout=f))
        with open("log/wavelet_" + "lr_" + str(lr) + '_gamma_' + str(gamma)
                  + '_' + time_str + ".txt", "w") as f:
            jobs.append(subprocess.Popen(['python', '../mnist_pool.py', '--lr', lr,
                                          '--gamma', gamma, '--pooling_type', 'scaled_adaptive_wavelet',
                                          '--tensorboard'], stdout=f))

        for job in jobs:
            job.wait()
