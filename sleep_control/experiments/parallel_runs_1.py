import os
from time import sleep
from datetime import datetime
from multiprocessing import Pool
import subprocess

previous_pid = 14516

prefix = ('python /home/admin-326/ipython-notebook/dqn4wirelesscontrol/'
          'sleep_control/experiments/')

cmd_list = [prefix+'experiment_QNN_Jan31_2240_batchnorm.py ' + str(i) for i in range(14)]

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



while(True):
    if previous_pid is not None and check_pid(previous_pid):
        print "Proceses {} is running, retry in 60 seconds.".format(previous_pid)
        sleep(60)
    else:
        break    

pool = Pool(7)
pool.map(run, cmd_list)
pool.close()
