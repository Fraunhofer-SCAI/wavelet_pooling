
import datetime

import subprocess
subprocess.call('pwd')

print('running mnist parameter sweep in parallel')


lr_lst = ['0.8', '0.65', '0.5', '0.25', '0.1']
gamma_lst = ['1.0', '0.99', '0.98', '0.95', '0.925', '0.9']

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
                                          '--gamma', gamma, '--pooling_type', 'wavelet',
                                          '--tensorboard'], stdout=f))

        for job in jobs:
            job.wait()
