#block(name=mp, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.01 --tensorboard --pooling_type max
#block(name=mp, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.01 --tensorboard --pooling_type max