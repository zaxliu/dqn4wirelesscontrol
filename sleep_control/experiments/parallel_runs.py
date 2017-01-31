from multiprocessing import Pool
import subprocess
prefix = ('python /home/admin-326/ipython-notebook/dqn4wirelesscontrol/'
          'sleep_control/experiments/')
def run(cmd):
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
    return

cmd_list = [prefix+'experiment_QNN_Jan31_1154_phi15.py ' + str(i) for i in range(10)]
cmd_list += [prefix+'experiment_QNN_Jan31_1156_phi25.py ' + str(i) for i in range(10)]

pool = Pool(7)
pool.map(run, cmd_list)
pool.close()
