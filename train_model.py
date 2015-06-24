###############################################################################
#                  Author: Markus Spanring HEPHY Vienna 2015                  #
###############################################################################
#
# Specifies parameters for training-algorithm and NN architecture
# and trains model on given dataset
#
###############################################################################


import sys
import os
import theano
from ModelScore import *
import pylearn2
import pylearn2.models.mlp as mlp
import pylearn2.training_algorithms
import pylearn2.training_algorithms.sgd
import pylearn2.train
import pylearn2.costs
import pylearn2.termination_criteria
import matplotlib.pyplot as plt
import numpy as np
import pylearn2.datasets.higgs_dataset


def init_train(args, conf = {'batch_size':200,
                            'prop_decrease':0.001,
                            'in_N':5,
                            'max_epochs':100,
                            'min_lr':0.000001,
                            'momentum_final':0.99,
                            'ptype': 'mu'}, 
                    **kwargs):
    """
    ARGS
    #1    hex_mask            Applies mask on dataset (1 in mask drops feature)
    #2    seed                Randomizes dataorder wenn seed is not None
    #3    nhid                Number of hidden layers
    #4    nodes               Number of Neurons in each hidden layer
    #5    lrinit              Initial value for the learnrate
    #6    lrdecay             Decay rate of learnrate for every batch
    #7    momentum_init       Initial value of momentum
    #8    momentum_final      max value of momentum
    #9    momentum_saturate   Number of epochs for momentum to reach maximum
    #10    flag_reg            Specifies if weights an y is included in data
    #                         0b01 contains weight
    #                         0b10 contains y

    CONF
    #1    batch_size          gradient will be averaged over batch_size
    #2    prop_decrease       factor by which the monitoring channel value has 
                              decreased
    #3    in_N                in N epochs
    #4    max_epochs          max number of epochs to monitor
    #5    min_lr              min value to which learnrate can decrease
    #6    ptype               particle type [mu ,el, all]

    Creates MLP (Multilayer Perceptron) model with 'nhid' Layers and 
    'nodes' Nodes.
    Training Algorithm is SGD (Stochastic Gradient Descent) with 
    Momentum, Learnrate Decay

    RETURN:

    'train'-object
    This object represents the complete model and algorithm
    .main_loop() starts training of this model
    """
    dev_path = os.environ['DNN_PATH']


    # Interpret arguments.
    hex_mask,seed, nhid, nodes, lrinit, lrdecay, momentum_init,\
    momentum_saturate,momentum_final,flag_reg = args

    hex_mask = str(hex_mask)
    if seed is not None:   
        seed = int(seed)
    else: seed = "None"  #Necessary to create modelname string correct

    batch_size = conf['batch_size']
    prop_decrease = conf['prop_decrease']
    in_N = conf['in_N']
    max_epochs = conf['max_epochs']
    min_lr = conf['min_lr']
    ptype = conf['ptype']

    stop = kwargs.get('stop',np.inf)
    left_slope = float( kwargs.get('left_slope',0.0) )


    modelname = '%s_%d_%d_%d_%0.14f_%0.14f_%0.6f_%d_%0.6f' % tuple(args[:-1])
    modelname = '%s_%s_%d' % (str( conf['ptype'] ),modelname,
                              int( conf['batch_size']) )
    if seed == "None":
        seed = None

    if not os.path.exists("%s/model" % dev_path):
        os.makedirs("%s/model" % dev_path)
        os.chown("%s/model" % dev_path, 1000,1000)

    if not os.path.exists("%s/log" % dev_path):
        os.makedirs("%s/log" % dev_path)
        os.chown("%s/log" % dev_path, 1000,1000)

    save_path = '%s/model/%s.pkl' % (dev_path, modelname)
    logfile = '%s/log/%s.log' % (dev_path, modelname)
    
    sys.stdout = sys.__stdout__
    print 'Using=%s' % theano.config.device # Can use gpus. 
    print 'Writing to %s' % logfile
    print 'Writing to %s' % save_path
    sys.stdout = open(logfile, 'w')

    # Dataset
    dataset_train = \
    pylearn2.datasets.higgs_dataset.DATASET(which_set='train',
                                      ptype = ptype,
                                      flag_reg = flag_reg,
                                      hex_mask = hex_mask,
                                      seed = seed,
                                      stop = stop)


    dataset_valid = \
    pylearn2.datasets.higgs_dataset.DATASET(which_set='valid',
                                      ptype = ptype,
                                      flag_reg = flag_reg,
                                      hex_mask = hex_mask,
                                      seed = seed)

    # dataset_test = \
    # pylearn2.datasets.higgs_dataset.DATASET(which_set='test',
    #                                   ptype = ptype,
    #                                   flag_reg = flag_reg,
    #                                   hex_mask = hex_mask,
    #                                   seed = seed)


    # Model
    nvis = dataset_train.X.shape[1] 
    istdev = 1.0/np.sqrt(nodes)
    layers = []

    for i in xrange(int(nhid)):
        # Hidden layer i
        print i,nodes
        layer = mlp.Tanh(layer_name = 'h%d' % i,
                                 dim=int(nodes),
                                 #left_slope = left_slope,
                                 istdev = (istdev if i>0 else 0.01) 
                                )
        layers.append(layer)

    layers.append(mlp.Sigmoid(layer_name='y',
                              dim=1,
                              irange=0.01,
                              monitor_style = 'bit_vector_class'))

    model = pylearn2.models.mlp.MLP(layers, nvis=nvis, seed=seed)

    # Cost
    cost = pylearn2.costs.mlp.Default() # Default cost.
    
    # Algorithm
    algorithm = pylearn2.training_algorithms.sgd.SGD(

            batch_size=int(batch_size),
            learning_rate=float(lrinit),
            learning_rule=pylearn2.training_algorithms.learning_rule.Momentum(
                                init_momentum = float(momentum_init),
                            ),
            monitoring_dataset = {#'train':dataset_train_monitor,                                  
                                  # 'test':dataset_test,
                                  'valid':dataset_valid
                                 },
            termination_criterion=  pylearn2.termination_criteria.And(criteria=[
                                    pylearn2.termination_criteria.MonitorBased(
                                        channel_name="valid_objective",
                                        prop_decrease = prop_decrease,
                                        N = in_N),
                                    pylearn2.termination_criteria.EpochCounter(
                                        max_epochs = max_epochs)
                                    ]),
            cost=cost,
                       
            update_callbacks=pylearn2.training_algorithms.sgd.ExponentialDecay(
                                decay_factor = float(lrdecay),  
                                min_lr = min_lr
                             )
                )

# 

    # Extensions 
    extensions=[ 
        pylearn2.training_algorithms.learning_rule.MomentumAdjustor(
            start=0,
            saturate = int(momentum_saturate), 
            final_momentum = float(momentum_final),
            )

        ]
    # Train
    train = pylearn2.train.Train(dataset=dataset_train,
                                 model=model,
                                 algorithm=algorithm,
                                 extensions=extensions,
                                 #Save to .pkl File
                                 save_path=save_path,
                                 save_freq=max_epochs)

    return train




def Compute_Objective(args,conf, **kwargs): 

        '''Initializes neural network with specified arguments and 
           configuration and trains.
        '''    
        # Train network.
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        dev_path = os.environ['DNN_PATH']

        modelname = '%s_%d_%d_%d_%0.14f_%0.14f_%0.6f_%d_%0.6f' % tuple(args[:-1])
        modelname = '%s_%s_%d' % (str( conf['ptype'] ),modelname,
                                        int( conf['batch_size']) )


        train = init_train(args,conf,**kwargs)
        train.main_loop()
        model = train.model

        sys.stdout = old_stdout
        sys.stderr = old_stderr
         
        # Return objective to hyperopt.
        loss = train.model.monitor.channels['valid_objective'].val_record

        try:
            fig = plt.figure()
            fig.suptitle('Progress of Training', fontsize=20)
            plt.xlabel('Number of Epochs')
            plt.ylabel('Value from Loss-function')
            plt.plot(loss)
            plt.plot(loss,'b.')
            plt.savefig('%s/figure/objective/%s.pdf' % (dev_path,modelname))
            plt.close('all')
            np.savetxt('%s/figure/objective/%s.dat' % (dev_path,modelname), loss)
        except:
            pass
           
        return loss[-1]

def f32(flt):
    if type(flt) is list or\
       type(flt) is np.ndarray:
        for i,arg in enumerate(flt):            
                flt[i] = f32(arg)

    elif type(flt) is float or \
         type(flt) is np.float64:

        flt = float(np.float32(flt)) 
    return flt

if __name__=='__main__':

    print "running as main"
    #ARGS
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
    #CONF
    #1    batch_size          gradient will be averaged over batch_size
    #2    prop_decrease       factor by which the channel value has decreased
    #3    in_N                in N epochs
    #4    max_epochs          max number of epochs to monitor
    #5    min_lr              min value to which learnrate can decrease
    #6    momentum_final      max value of momentum
    #7    ptype               particle type [mu ,el, all]



    hex_mask = '0x0000'
    seed = 74             
    nhid = 2
    nodes = 25              
    lrinit = f32( 0.00300000002608 )            
    lrdecay = f32(1.00007057189941)            
    momentum_init =  f32(0.850000)
    momentum_final = f32(0.990000)
    momentum_saturate = 100             
          


    dev_path = os.environ['DNN_PATH']

    flag_reg = 0xb

    arg = [hex_mask,seed,nhid,nodes,lrinit,lrdecay,\
            momentum_init,momentum_saturate,momentum_final,flag_reg]

    conf = {'batch_size':100,
        'prop_decrease':0.0005,
        'in_N':100,
        'max_epochs':1000,
        'min_lr':1e-7,
        'ptype': 'mu'}

    Compute_Objective( args = arg, conf = conf )

    modelname = '%s_%d_%d_%d_%0.14f_%0.14f_%0.6f_%d_%0.6f' % tuple(arg[:-1])
    modelname = '%s_%s_%d.pkl' % (str( conf['ptype'] ),modelname,
                                  int( conf['batch_size']) )

    


    AMS = ModelAMS(modelname = modelname,
                   ptype = conf['ptype'],
                   seed = seed,
                   flag_reg = flag_reg
                   )

    score = AMS.ams_model()
    print score

