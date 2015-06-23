###############################################################################
#                  Author: Markus Spanring HEPHY Vienna 2015                  #
###############################################################################
#
# Optimizes model specified in 'train_model.py' with spearmint.
# Parameters to optimize are defined in 'config.json'
#
###############################################################################


import train_model
import os
import random
from spr_logtools import *
from ModelScore import *




    #1    hex_mask            Applies mask on dataset (1 in mask drops feature)
    #2    seed                Randomizes dataorder wenn seed is not None
    #3    nhid                Number of hidden layers
    #4    nodes               Number of Neurons in each hidden layer
    #5    lrinit              Initial value for the learnrate
    #6    lrdecay             Decay rate of learnrate for every batch
    #7    momentum_init       Initial value of momentum
    #8    momentum_saturate   Number of epochs for momentum to reach maximum
    #11   flag_reg            Specifies if weights an y is included in data
    #                         0b001 contains weight
    #                         0b010 contains y
    #                         0b100 use bottleneck

    #1    batch_size          gradient will be averaged over batch_size
    #2    prop_decrease       factor by which the channel value has decreased
    #3    in_N                in N epochs
    #4    max_epochs          max number of epochs to monitor
    #5    min_lr              min value to which learnrate can decrease
    #6    momentum_final      max value of momentum
    #7    ptype               particle type [mu ,el, all]


def cast(args):
    '''Casts the correct datatype on args
    '''
    type_list = [str,int,int,int,float,float,float,int,float,int]
    for i,t in enumerate(type_list):
        args[i] = t(args[i])
    
    args = f32(args)
    return args

def optimize_DNN(params, s_name, maxScore, fixed_param = {}):

    """
    Reads all parameters 'param' to optimize.
    If certain dictionary entries are not 
    available in 'params' a standard value 
    is set for this hyperparameter.
    Before each training it checks 
    if the configuration was already calculated
    by calling the 'check_duplicate'-function
    """

    if fixed_param is None: fixed_param = {}
    dev_path = os.environ['DEV_PATH']
    flag_reg = 0x3
    seed = 74   
    threshold = 2.0
    ERROR = False
    loss = 1.

    args = cast(['0x0000',seed,5,500,-2,-5,\
                0.8,300,0.99,flag_reg])

    conf = {'batch_size':70,
        'prop_decrease':0.0005,
        'in_N':15,
        'max_epochs':800,
        'min_lr':0.000001,
        'ptype': 'mu'}


    stop = fixed_param.get('stop',np.inf)
    left_slope = fixed_param.get('left_slope',0.0)

    conf_param = ['batch_size', 'prop_decrease',
                  'in_N','max_epochs','min_lr','ptype']

    for param in iter(conf_param):
        if fixed_param.has_key(param):
            conf[param] = fixed_param[param]

        elif params.has_key(param):
            conf[param] = params[param]


    hyper_param = ['mask', 'seed', 'nhid','nodes',
                   'lrinit','lrdecay','momentum_init',
                   'momentum_saturate','momentum_final']

    for i, param in enumerate( hyper_param ):
        if fixed_param.has_key(param):  
                args[i] = f32(fixed_param[param])

        elif params.has_key(param):
                args[i] = f32(params[param])

    if args[4] < 0:
        args[4] = f32(10**args[4])
    if args[5] < 0:
        args[5] = f32(1+10**args[5])



    dup = check_duplicate(args = args, conf= conf, s_name = s_name) 

    if dup[0]:
        print 'already got that entry:\
               %s_%d_%d_%d_%0.14f_%0.14f_%0.6f_%d_%0.6f.pkl. Using new seed' %\
               tuple(args[:-1])

        args[1] = int(random.random() * 100)


    modelname = '%s_%d_%d_%d_%0.14f_%0.14f_%0.6f_%d_%0.6f' % tuple(args[:-1])
    modelname = '%s_%s_%d.pkl' % (str( conf['ptype'] ),modelname,
                                  int( conf['batch_size']) )

    print 'start training'
    loss = train_model.Compute_Objective(args = args, conf = conf,
                                             stop = stop, left_slope = left_slope)        

    if not ERROR:       
        AMS = ModelAMS(modelname = modelname,
                       ptype = conf['ptype'],
                       seed = args[1],
                       flag_reg = args[9]
                       )

        score = AMS.ams_model()
    else:
        score = ",".join(["0","0","0","0",
                          modelname.replace(".pkl","").replace("_",","),'0'])

    print 'Loss:',           str(loss)
    print "score:%s" % score
    print "conf-begin-"
    print "prop_decrease:",  conf['prop_decrease']
    print "in_N:",           conf['in_N']
    print "max_epochs:",     conf['max_epochs']
    print "min_lr:",         conf['min_lr']
    print 'flag_reg:',        hex(flag_reg)
    print "conf-end-"
    
    if ERROR:
        print 'removing because of error...'
        if os.path.exists("%s/model/%s" % (dev_path, modelname) ):
            os.remove("%s/model/%s" % (dev_path, modelname))

        if os.path.exists( "%s/log/%s"% \
           (dev_path ,modelname.replace(".pkl",".log") ) ):

            os.remove( "%s/log/%s"%\
            (dev_path ,modelname.replace(".pkl",".log")) )
        return 1



    if float(AMS.score['test_rb']) >= maxScore:  

        if float(AMS.score['test_rb']) > threshold:
            AMS.plot_hist()


    if float(AMS.score['test_rb']) < threshold:
        print 'Model not good enough (threshold is set to %s)...deleting' %\
               str(threshold)
        os.remove("%s/model/%s" % (dev_path, modelname))
        os.remove("%s/log/%s"% (dev_path ,modelname.replace(".pkl",".log")) )


    return float(AMS.score['test_rb'])*-1


def main(job_id, params,**kwargs): 


    s_name = kwargs.get('s_name','smry-file.dat')
    fixed_param = kwargs.get('fixed_param',{})

    maxScore = write_summary(s_name = s_name)

    result = optimize_DNN( params = params, s_name = s_name,
                           maxScore = maxScore, fixed_param = fixed_param)   

    return result