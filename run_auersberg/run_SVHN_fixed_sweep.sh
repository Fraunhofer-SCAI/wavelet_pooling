#block(name=fwplr08m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.08 --tensorboard --pooling_type wavelet --momentum 0.0 --epochs 50 --data_set SVHN

#block(name=fwplr08m4, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.08 --tensorboard --pooling_type wavelet --momentum 0.4 --epochs 50 --data_set SVHN

#block(name=fwplr08m6, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.08 --tensorboard --pooling_type wavelet --momentum 0.6 --epochs 50 --data_set SVHN

#block(name=fwplr08m8, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.08 --tensorboard --pooling_type wavelet --momentum 0.8 --epochs 50 --data_set SVHN

#block(name=fwplr06m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.06 --tensorboard --pooling_type wavelet --momentum 0.0 --epochs 50 --data_set SVHN

#block(name=fwplr06m04, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.06 --tensorboard --pooling_type wavelet --momentum 0.4 --epochs 50 --data_set SVHN

#block(name=fwplr06m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.06 --tensorboard --pooling_type wavelet --momentum 0.6 --epochs 50 --data_set SVHN

#block(name=fwplr06m08, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.06 --tensorboard --pooling_type wavelet --momentum 0.8 --epochs 50 --data_set SVHN

#block(name=fwplr04m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.04 --tensorboard --pooling_type wavelet --momentum 0.0 --epochs 50 --data_set SVHN

#block(name=fwplr04m04, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.04 --tensorboard --pooling_type wavelet --momentum 0.4 --epochs 50 --data_set SVHN

#block(name=fwplr04m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.04 --tensorboard --pooling_type wavelet --momentum 0.6 --epochs 50 --data_set SVHN

#block(name=fwplr04m08, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.04 --tensorboard --pooling_type wavelet --momentum 0.8 --epochs 50 --data_set SVHN

#block(name=fwplr02m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.02 --tensorboard --pooling_type wavelet --momentum 0.0 --epochs 50 --data_set SVHN

#block(name=fwplr02m04, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.02 --tensorboard --pooling_type wavelet --momentum 0.4 --epochs 50 --data_set SVHN

#block(name=fwplr02m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.02 --tensorboard --pooling_type wavelet --momentum 0.6 --epochs 50 --data_set SVHN

#block(name=fwplr02m08, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar_SVHN.py -- --lr 0.02 --tensorboard --pooling_type wavelet --momentum 0.8 --epochs 50 --data_set SVHN
