import os
from time import sleep
from datetime import datetime
from multiprocessing import Pool
import subprocess

previous_pid = None

prefix = ('python /home/admin-326/ipython-notebook/dqn4wirelesscontrol/'
          'sleep_control/experiments/')

cmd_list = [prefix+'experiment_QNN_Feb1_1740_phi15_dsy.py ' + str(i) for i in range(14)]
postproc_list = [prefix+'log_indexing_phiNN.py msg_QNN_Feb1_1740 14']
postproc_list += ['tar czf ./log/tarballs/msg_QNN_Feb1_1740_x14.tar.gz ./log/msg_QNN_Feb1_1740*.log']

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
pool.map(run, postproc_list)
pool.close()


