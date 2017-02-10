import os
import time
from time import sleep
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
import subprocess

previous_pid = None

prefix = ('python /home/admin-326/ipython-notebook/dqn4wirelesscontrol/'
          'sleep_control/experiments/')
runs = []
# -- 13 sim
exp_file = 'experiment_DynaQtable_Feb7_2228_13sim.py'
log_file = 'msg_DynaQtable_Feb7_2228'
num_sim = 13
num_log = 7
exp_list = [prefix + exp_file + ' ' + str(i) for i in range(num_log)]
cmd_index = prefix+'log_indexing_DynaQtable.py {log_file} {num_sim} {num_log}'.format(
    log_file=log_file, num_sim=num_sim, num_log=num_log)
cmd_tar = 'tar czf ./log/tarballs/{log_file}_x{num_log}.tar.gz ./log/{log_file}*.log'.format(
    log_file=log_file, num_log=num_log)
runs.append((exp_list, cmd_index, cmd_tar, (log_file, num_log, num_log)))
# -- 13 sim
exp_file = 'experiment_DynaQtable_Feb7_2229_17sim.py'
log_file = 'msg_DynaQtable_Feb7_2229'
num_sim = 17
num_log = 7
exp_list = [prefix + exp_file + ' ' + str(i) for i in range(num_log)]
cmd_index = prefix+'log_indexing_DynaQtable.py {log_file} {num_sim} {num_log}'.format(
    log_file=log_file, num_sim=num_sim, num_log=num_log)
cmd_tar = 'tar czf ./log/tarballs/{log_file}_x{num_log}.tar.gz ./log/{log_file}*.log'.format(
    log_file=log_file, num_log=num_log)
runs.append((exp_list, cmd_index, cmd_tar, (log_file, num_log, num_log)))

def check_pid(pid):        
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True

def run(cmd):
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
    return

def load_dataframes(prefix, n_run, n=None):
    if n is None:
        n = n_run
    files = [prefix + "_{}.log".format(i) for i in range(n)]
    file_list = ['/home/admin-326/ipython-notebook/dqn4wirelesscontrol/'
                 'sleep_control/experiments/log/index/'
                 + prefix +'_x{}/'.format(n_run) +'index_'+file+'.csv' for file in files]
    df_list = [None]*n
    for i in range(n):
        t = time.time()
        df = pd.read_csv(file_list[i], delimiter=';', index_col=0)
        df.loc[:, 'start_ts'] = df['start_ts'].apply(lambda x: pd.to_datetime(x))
        df.set_index('start_ts', inplace=True)
        df['total_reward'] = df['tr_reward'] + df['op_cost']
        df_list[i] = df
        print df.shape,
        print files[i],
        print "{:.2f} sec".format(time.time()-t)
    return df_list   

def get_step_reward(file_prefix, num_total, num_load):
    df_list = load_dataframes(file_prefix, num_total, num_load)
    # df_list = filter(lambda x: x.shape[0]==302400, df_list)
    # start = pd.to_datetime("2014-10-16 9:30:00")
    # end = pd.to_datetime("2014-10-21 9:30:00")
    start = pd.to_datetime("2013-10-16 9:30:00")
    end = pd.to_datetime("2015-10-21 9:30:00")
    delta = pd.Timedelta('2 seconds')

    step_reward = np.zeros(len(df_list))
    for i, df in enumerate(df_list):
        df = df.loc[start:end]
        print (i, df.shape[0])
        step = (df.index-df.index[0])/delta+1
        ts = df['total_reward'].cumsum()/step
        step_reward[i] = ts.iloc[-1]
    return step_reward

def log_step_reward(file, step_reward):
    with open('/home/admin-326/ipython-notebook/dqn4wirelesscontrol/'
              'sleep_control/experiments/log/'
              '{file}.reward'.format(file=file), 'w') as reward_file:
        print>>reward_file, step_reward.tolist()

while(True):
    if previous_pid is not None and check_pid(previous_pid):
        print datetime.now().strftime('[%Y-%m-%d %H:%M:%S]'),
        print "Proceses {} is running, retry in 600 seconds. (I'm {})".format(previous_pid, os.getpid())
        sleep(600)
    else:
        break    

pool = Pool(7)
for exp_list, cmd_index, cmd_tar, (log_file, num_log, num_log) in runs:
    pool.map(run, exp_list)
    run(cmd_index)
    run(cmd_tar)
    step_reward = get_step_reward(log_file, num_log, num_log)
    print step_reward
    log_step_reward(log_file, step_reward)
pool.close()

