import subprocess
subprocess.call('pwd')

print('running mnist exps.')


for exp in range(10):
    job = subprocess.Popen(['python', '../mnist_pool.py',
                            '--lr',  '0.12', '--tensorboard',
                            '--pooling_type', 'adaptive_wavelet',
                            '--momentum', '0.6',
                            '--gamma', '0.95', '--epochs', '25'])
    job.wait()

for exp in range(10):
    job = subprocess.Popen(['python', '../mnist_pool.py',
                            '--lr',  '0.12', '--tensorboard',
                            '--pooling_type', 'adaptive_max',
                            '--momentum', '0.6',
                            '--gamma', '0.95', '--epochs', '25'])
    job.wait()

for exp in range(10):
    job = subprocess.Popen(['python', '../mnist_pool.py',
                            '--lr',  '0.12', '--tensorboard',
                            '--pooling_type', 'adaptive_avg',
                            '--momentum', '0.6',
                            '--gamma', '0.95', '--epochs', '25'])
    job.wait()

for exp in range(10):
    job = subprocess.Popen(['python', '../mnist_pool.py',
                            '--lr', '0.12', '--tensorboard',
                            '--pooling_type', 'max',
                            '--momentum', '0.6',
                            '--gamma', '0.95', '--epochs', '25'])
    job.wait()

for exp in range(10):
    job = subprocess.Popen(['python', '../mnist_pool.py',
                            '--lr',  '0.12', '--tensorboard',
                            '--pooling_type', 'avg',
                            '--momentum', '0.6',
                            '--gamma', '0.95', '--epochs', '25'])
    job.wait()

for exp in range(10):
    job = subprocess.Popen(['python', '../mnist_pool.py',
                            '--lr',  '0.12', '--tensorboard',
                            '--pooling_type', 'scaled_wavelet',
                            '--momentum', '0.6',
                            '--gamma', '0.95', '--epochs', '25'])
    job.wait()

for exp in range(10):
    job = subprocess.Popen(['python', '../mnist_pool.py',
                            '--lr',  '0.12', '--tensorboard',
                            '--pooling_type', 'wavelet',
                            '--momentum', '0.6',
                            '--gamma', '0.95', '--epochs', '25'])
    job.wait()
