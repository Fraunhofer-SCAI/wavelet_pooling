import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


def tensoboard_average(y, window):
    '''
    * The smoothing algorithm is a simple moving average, which, given a
     * point p and a window w, replaces p with a simple average of the
     * points in the [p - floor(w/2), p + floor(w/2)] range.
    '''
    window_vals = []
    length = y.shape[-1]
    for p_no in range(0, length, window):
        if p_no > window/2 and p_no < length - window/2:
            window_vals.append(np.mean(y[p_no-int(window/2):p_no+int(window/2)]))
    return np.array(window_vals)


def return_logs(path, window_size=0, vtag='mse'):
    """
    Load loss values from logfiles smooth and return.
    """
    dir_lst = []
    file_lst = []
    for root, dirs, files in os.walk(path):
        print(dirs)
        if len(dirs) > 0:
            dir_lst.extend(dirs)
        else:
            file_lst.append(os.path.join(root, files[0]))
    xy_lst = []
    for no, p in enumerate(file_lst):
        adding_umc = []
        try:
            for e in tf.compat.v1.train.summary_iterator(p):
                for v in e.summary.value:
                    if v.tag == vtag:
                        # print(v.simple_value)
                        adding_umc.append(v.simple_value)
        except:
            # ingnore that silly data loss error....
            pass
        # x = np.array(range(len(adding_umc)))

        y = np.array(adding_umc)
        if window_size > 1:
            yhat = tensoboard_average(y, window_size)
        else:
            yhat = y
        xhat = np.linspace(0, y.shape[0], yhat.shape[0])
        xy_lst.append([[xhat, yhat], p])
    return xy_lst


if __name__ == '__main__':
    logdir = './experiments/'
    logs = return_logs(path=logdir, vtag='test_accuracy')
