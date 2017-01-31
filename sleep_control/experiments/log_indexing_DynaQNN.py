# coding: utf-8

from sys import argv
import os
import regex as re
import pandas as pd
from multiprocessing import Pool


def msg_sim(num_sim):
    # RegEx format template for a single simulation round. Group name suffixed by sim_num
    msg_single_sim = (
    # action: only apply uniform random policy
    '    QAgent.act_\(\): randomly choose action \(None state\).\n'
    # simulated experience
    '    DynaMixin.reinforce_\(\): simulated experience \d: [\w\(\)\.\-, \']+\n'
    # QAgent reinforce
    '(?:(?:    QAgentNN.reinforce_\(\): (?:'
            '(?:last_state is None.)|'
            '(?:last_reward is None.)|'
            '(?:state is None.)|'
            '(?:unfull memory.)'
    ')\n)|(?:'
    '(?:    QAgentNN.reinforce_\(\): update counter (?P<counter_update_sim{suffix}>\d+), freeze counter (?P<counter_freeze{suffix}>\d+), rs counter (?P<counter_rs{suffix}>\d+).\n)'
    '(?:'
        '(?:    QAgentNN.reinforce_\(\): update loss is (?P<loss{suffix}>[\w\.-]+), reward_scaling is (?P<reward_scaling{suffix}>[\w\.-]+)\n)'
    # mini-batch distribution: wake and sleep (float or string)
        '(?:        QAgentNN.reinforce_\(\): batch action distribution: (\{{'
                        '\(False, \'serve_all\'\): (?P<batch_dist_wake{suffix}>[\w\.-]+), '
                        '\(True, None\): (?P<batch_dist_sleep{suffix}>[\w\.-]+)'
        '\}})\n)'
    ')?'
    '))'
    )
    return ''.join([msg_single_sim.format(suffix='_sim_'+str(n)) for n in range(num_sim)])

def epoch_msg(num_sim):
    return (
    # epoch: uint
    # time stamp: YYYY-MM-DD HH:MM:SS
    'Epoch (?P<epoch>\d+), (?P<start_ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (?P<end_ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\n'
    # last reward: float or None
    'Emulation.step\(\): last reward: (?:(?P<last_reward>[-]*[\d\.]+)|(?:None))\n'

    # Sessions: uint
    '    TrafficEmulator.generate_traffic\(\): located (?P<session_in>\d+), droped (?P<session_out>\d+), left (?P<session_net>\d+) sessions.\n'
    # Requests: uint
    '        TrafficEmulator.generate_requests_\(\): generated (?P<req_generated>\d+) requests.\n'

    # Observation (uint, uint, uint)
    'Emulation.step\(\): observation: \((?P<ob_last_q>\d+), (?P<ob_last_t>\d+), (?P<ob_new_q>\d+)\)\n'

    # Environment model: unfull tr win warning or stride and eval counters (uint, uint)
    # eval score
    '    SJTUModel.improve\(\): (?:unfull traffic window.|stride counter (?P<counter_model_stride>\d+), eval counter (?P<counter_model_eval>\d+))\n'
    '(?:    SJTUModel.improve\(\): model score (?P<score_model>[\w\.-]+), expected (?P<score_model_exp>[\w\.-]+), diff. (?P<score_model_diff>[\w\.-]+)\n)?'
    '    DynaMixin.improve_translate_\(\): belief state (?:None|\((?P<belief_last_t>[\w\.-]+), \d+, (?:True|False)\))\n'

    # agent update msg: 4 strings or loss (float or string) + rs (float or string)
    '(?:(?:    QAgentNN.reinforce_\(\): (?P<agent_update_msg>'
            '(?:last_state is None.)|'
            '(?:last_reward is None.)|'
            '(?:state is None.)|'
            '(?:unfull memory.)'
    ')\n)|(?:'
    '(?:    QAgentNN.reinforce_\(\): update counter (?P<counter_update>\d+), freeze counter (?P<counter_freeze>\d+), rs counter (?P<counter_rs>\d+).\n)'
    '(?:'
        '(?:    QAgentNN.reinforce_\(\): update loss is (?P<loss>[\w\.-]+), reward_scaling is (?P<reward_scaling>[\w\.-]+)\n)'
    # mini-batch distribution: wake and sleep (float or string)
        '(?:        QAgentNN.reinforce_\(\): batch action distribution: (?P<batch_dist>\{'
                        '\(False, \'serve_all\'\): (?P<batch_dist_wake>[\w\.-]+), '
                        '\(True, None\): (?P<batch_dist_sleep>[\w\.-]+)'
        '\})\n)'
    ')?'
    '))'
    ) + \
    msg_sim(num_sim) + \
    (
    # action msg: random or policy
    #   q_values if epsilon greedy
    # policy msg
    '    QAgent.act_\(\): '
        '(?P<agent_act_msg>('
            '(?:randomly choose action)|'
            '(?:choose best q among '
                '(?P<q_vals>\{\(False, \'serve_all\'\): (?P<q_wake>[\w\.-]+), \(True, None\): (?P<q_sleep>[\w\.-]+)\}))'
        ')'
        ' \((?P<agent_act_basis>[a-zA-Z ]+)\)'
        ').\n'

    # agent action: (True, None) or (False, 'serve_all')
    # agent update: [ignore]
    'Emulation.step\(\): control: (?P<agent_action>\([a-zA-Z,_ \']+\)), agent update: .*\n'

    # Service: 
    #   req: served, queued, rejected (retried+canceled), unattended [uint]
    #   reward: service, wait, fail [int]
    #   buffer: pending, waiting, served, failed
    '        TrafficEmulator.evaluate_service_\(\): '
                'served (?P<req_served>\d+), queued (?P<req_queued>\d+), '
                'rejected (?P<req_rejected>\d+) \((?P<req_retried>\d+), (?P<req_canceled>\d+)\), unattended (?P<req_unattended>\d+), '
                'reward ([-]?[\d\.]+) \((?P<tr_reward_serve>[-]?[\d\.]+), (?P<tr_reward_wait>[-]?[\d\.]+), (?P<tr_reward_fail>[-]?[\d\.]+)\)\n'
    '        TrafficEmulator.evaluate_service_\(\): '
                'pending (?P<req_pending_all>\d+), waiting (?P<req_waiting_all>\d+), '
                'served (?P<req_served_all>\d+), failed (?P<req_failed_all>\d+)\n'

    # # operation cost: float
    # # traffic reward: float
    'Emulation.step\(\): cost: (?P<op_cost>[-]*[\d\.]+), reward: (?P<tr_reward>[-]*[\d\.]+)'
    # # last line
    '\n{0,1}'
    )

    
if len(argv)>2:
    file_prefix = argv[1]
    num_sim = int(argv[2])
    num = int(argv[3])
    files =[
        file_prefix + "_{}.log".format(i) for i in range(num)
    ]
    print "Processing {} files for {} sims, total {}".format(argv[1], argv[2], argv[3])
    
    re_epoch_msg = re.compile(epoch_msg(num_sim))
    
    index_dir = file_prefix+'_x{}/'.format(num)
    if not os.path.exists('./log/index/'+index_dir):
        os.mkdir('./log/index/'+index_dir)
        print 'Created index dir {}'.format(index_dir)
    else:
        print 'Index dir already exist'

    
    def index_file(file):
        if os.path.isfile("./log/index/"+index_dir+"index_"+file+".csv"):
            print 'file {} already exist!'.format("./log/index/"+index_dir+"index_"+file+".csv")
        else:
            with open('./log/'+file, "r") as f_log:
                all_log = "".join(f_log.readlines()).split('\n\n')
                extract = [re_epoch_msg.search(piece) for piece in all_log]
                df = pd.DataFrame.from_dict([piece.groupdict() for piece in extract if piece is not None])
                df.set_index('epoch')
                df.index.name = 'epoch'
            with open("./log/index/"+index_dir+"index_"+file+".csv", "w") as f_ind:
                df.to_csv(f_ind, sep=';', index=True, header=True)
            print (file, df.shape)
        return
        
    p = Pool(3)
    p.map(index_file, files)
else:
    print "Please specify pattern of log file name and num of files."




