from multiprocessing import Pool
import subprocess
prefix = ('python /home/admin-326/ipython-notebook/dqn4wirelesscontrol/'
          'sleep_control/experiments/')
def run(cmd):
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
    return

cmd_list = [prefix+'experiment_QNN_Jan25_2319.py ' + str(i) for i in range(24)]
cmd_list += [prefix+'experiment_DynaQNN_Jan25_2320.py ' + str(i) for i in range(24)]

pool = Pool(6)
pool.map(run, cmd_list)
pool.close()
