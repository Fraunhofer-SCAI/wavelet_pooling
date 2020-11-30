import subprocess
subprocess.call('pwd')

print('running mnist parameter sweep in parallel')

for exp in range(10):
    with open("p_" + 'max_' + str(exp)) as f:
            job = subprocess.Popen(['python', '../mnist_pool.py',
                                    '--lr 0.12', '--tensorboard',
                                    '--pooling_type max',
                                    '--momentum 0.6',
                                    '--gamma 0.95', '--epochs 25')])

for exp in range(10):
    with open("p_" + 'mean_'  + str(exp)) as f:
            job = subprocess.Popen(['python', '../mnist_pool.py',
                                    '--lr 0.12', '--tensorboard',
                                    '--pooling_type mean',
                                    '--momentum 0.6',
                                    '--gamma 0.95', '--epochs 25')])