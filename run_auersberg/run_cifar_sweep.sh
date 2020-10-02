#block(name=awplr08m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type adaptive_wavelet --momentum 0.0

#block(name=awplr08m4, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type adaptive_wavelet --momentum 0.4

#block(name=awplr08m6, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type adaptive_wavelet --momentum 0.6

#block(name=awplr08m8, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type adaptive_wavelet --momentum 0.8

#block(name=awplr06m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type adaptive_wavelet --momentum 0.0

#block(name=awplr06m04, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type adaptive_wavelet --momentum 0.4

#block(name=awplr06m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type adaptive_wavelet --momentum 0.6

#block(name=awplr06m08, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type adaptive_wavelet --momentum 0.8

#block(name=awplr04m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.04 --tensorboard --pooling_type adaptive_wavelet --momentum 0.0

#block(name=awplr04m44, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.04 --tensorboard --pooling_type adaptive_wavelet --momentum 0.4

#block(name=awplr04m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.04 --tensorboard --pooling_type adaptive_wavelet --momentum 0.6

#block(name=awplr02m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type adaptive_wavelet --momentum 0.0

#block(name=awplr02m04, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type adaptive_wavelet --momentum 0.4

#block(name=awplr02m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type adaptive_wavelet --momentum 0.6

#block(name=awplr02m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type adaptive_wavelet --momentum 0.8



#block(name=swplr08m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type scaled_wavelet --momentum 0.0

#block(name=swplr08m4, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type scaled_wavelet --momentum 0.4

#block(name=swplr08m6, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type scaled_wavelet --momentum 0.6

#block(name=swplr08m8, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type scaled_wavelet --momentum 0.8

#block(name=swplr06m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type scaled_wavelet --momentum 0.0

#block(name=swplr06m04, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type scaled_wavelet --momentum 0.4

#block(name=swplr06m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type scaled_wavelet --momentum 0.6

#block(name=swplr06m08, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type scaled_wavelet --momentum 0.8

#block(name=swplr04m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.04 --tensorboard --pooling_type scaled_wavelet --momentum 0.0

#block(name=swplr04m44, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.04 --tensorboard --pooling_type scaled_wavelet --momentum 0.4

#block(name=swplr04m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.04 --tensorboard --pooling_type scaled_wavelet --momentum 0.6

#block(name=swplr02m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type scaled_wavelet --momentum 0.0

#block(name=swplr02m04, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type scaled_wavelet --momentum 0.4

#block(name=swplr02m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type scaled_wavelet --momentum 0.6

#block(name=swplr02m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type scaled_wavelet --momentum 0.8



#block(name=maxplr08m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type max --momentum 0.0

#block(name=maxplr08m4, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type max --momentum 0.4

#block(name=maxplr08m6, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type max --momentum 0.6

#block(name=maxplr08m8, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type max --momentum 0.8

#block(name=maxplr06m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type max --momentum 0.0

#block(name=maxplr06m04, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type max --momentum 0.4

#block(name=maxplr06m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type max --momentum 0.6

#block(name=maxplr06m08, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type max --momentum 0.8

#block(name=maxplr04m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.04 --tensorboard --pooling_type max --momentum 0.0

#block(name=maxplr04m44, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.04 --tensorboard --pooling_type max --momentum 0.4

#block(name=maxplr04m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.04 --tensorboard --pooling_type max --momentum 0.6

#block(name=maxplr02m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type max --momentum 0.0

#block(name=maxplr02m04, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type max --momentum 0.4

#block(name=maxplr02m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type max --momentum 0.6

#block(name=maxplr02m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type max --momentum 0.8



#block(name=avgplr08m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type avg --momentum 0.0

#block(name=avgplr08m4, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type avg --momentum 0.4

#block(name=avgplr08m6, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type avg --momentum 0.6

#block(name=avgplr08m8, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.08 --tensorboard --pooling_type avg --momentum 0.8

#block(name=avgplr06m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type avg --momentum 0.0

#block(name=avgplr06m04, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type avg --momentum 0.4

#block(name=avgplr06m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type avg --momentum 0.6

#block(name=avgplr06m08, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.06 --tensorboard --pooling_type avg --momentum 0.8

#block(name=avgplr04m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.04 --tensorboard --pooling_type avg --momentum 0.0

#block(name=avgplr04m44, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.04 --tensorboard --pooling_type avg --momentum 0.4

#block(name=avgplr04m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.04 --tensorboard --pooling_type avg --momentum 0.6

#block(name=avgplr02m00, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type avg --momentum 0.0

#block(name=avgplr02m04, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type avg --momentum 0.4

#block(name=avgplr02m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type avg --momentum 0.6

#block(name=avgplr02m06, threads=2, memory=7500, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ../train_cifar.py -- --lr 0.02 --tensorboard --pooling_type avg --momentum 0.8