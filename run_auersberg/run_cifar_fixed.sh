#block(name=fwp, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.1 --tensorboard --pooling_type wavelet --momentum 0.9  --model DenseNet