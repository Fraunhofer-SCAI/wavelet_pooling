import subprocess
subprocess.call('pwd')

print('running mnist extra sep non-sep rebuttal exps.')

for exp in range(10):
    job = subprocess.Popen(['python', '../mnist_pool.py',
                            '--lr',  '0.12', '--tensorboard',
                            '--pooling_type', 'seperable_wavelet',
                            '--momentum', '0.6',
                            '--gamma', '0.95', '--epochs', '25'])
    job.wait()

for exp in range(10):
    job = subprocess.Popen(['python', '../mnist_pool.py',
                            '--lr', '0.12', '--tensorboard',
                            '--pooling_type', 'wavelet',
                            '--momentum', '0.6',
                            '--gamma', '0.95', '--epochs', '25'])
    job.wait()
