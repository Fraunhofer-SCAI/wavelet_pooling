#block(name=meanp, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.1 --tensorboard --pooling_type avg --momentum 0.9 --model DenseNet --wd 1e-4 --nesterov
