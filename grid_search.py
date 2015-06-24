###############################################################################
#                  Author: Markus Spanring HEPHY Vienna 2015                  #
###############################################################################
#
# Performs brute force grid search. The same grid is calculated
# for commbinations of parameters defined in layer_lst
#
###############################################################################
import numpy as np
import time
import sys
import os
import train_model
from ModelScore import *


def main():
    
    dev_path = os.environ['DEV_PATH']

    hex_mask = '0x0000'
    seed = 42
    nhid = 5
    nodes = 500
    lrinit = 0.000001
    lrdecay = 1.001
    momentum_init = 0.5
    momentum_saturate = 50
    momentum_final = 0.99
    flag_reg = 0x3

    args = [hex_mask,seed,nhid,nodes,lrinit,lrdecay,\
            momentum_init,momentum_saturate,momentum_final,flag_reg]

    conf = {'batch_size':70,
        'prop_decrease':0.0005,
        'in_N':15,
        'max_epochs':800,
        'min_lr':0.000001,
        'ptype': 'mu'}

    lrinit_lst  = log_spacing(0.03, 0.00001, steps=20)
    lrd_lst = 1+log_spacing(0.001, 0.000001, steps=20)

    grid = get_grid( x=lrinit_lst, y=lrd_lst,
                        x_key = 'lrinit', y_key = 'lrdecay') 

    nhid_lst = np.array([5])
    sat_lst = np.array([20,50,100,200])

    layer_lst = get_grid(x=nhid_lst, y=sat_lst,
                         x_key = 'nhid', y_key = 'momentum_saturate')


    for layer in layer_lst:
        count = 0
        sname = get_sname()        
        loss = 0
        scores = []
        for gr in grid:
            count += 1
            stime = time.time()
            args = get_new_args(args, layer, gr)

            ERROR = False
            try:
                loss = train_model.Compute_Objective(args = args,
                                                     conf = conf,
                                                     left_slope = 0.1)
            except KeyboardInterrupt:
                write_summary(scores, sname = sname)
                sys.exit()

            except Exception,e:                
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                print e
                ERROR = True

            etime = time.time() - stime

            modelname = '%s_%d_%d_%d_%0.14f_%0.14f_%0.6f_%d' % tuple(args[:-1])
            modelname = '%s_%s_%0.6f_%d.pkl' % (str( conf['ptype'] ),modelname,
                                         float( conf['momentum_final'] ),
                                         int( conf['batch_size']) )

            if not ERROR:       
                AMS = ModelAMS(modelname = modelname,
                               ptype = conf['ptype'],
                               seed = args[1],
                               flag_reg = args[8]
                               )

                score = '%s,%s,%s,%s\n' % (str(count),
                                         AMS.ams_model(),
                                         str(etime),
                                         str(loss) )
                print '-'*40
                print score
                print '-'*40
            else:
                score =  '%s,%s,%s,%s,%s\n' %\
                         ( str(count), 
                           ','.join(['0']*4),
                           modelname.replace('.pkl', '').replace('_',','),
                           str(etime),
                           str(loss) )


            scores.append( score )

            if os.path.exists("%s/model/%s" % (dev_path, modelname) ):
                os.remove("%s/model/%s" % (dev_path, modelname))
            if os.path.exists( "%s/log/%s"% (dev_path , modelname.replace(".pkl",".log") ) ):
                os.remove( "%s/log/%s"% (dev_path, modelname.replace(".pkl",".log")) )

            write_summary(scores, sname = sname)


def write_summary(scores, sname = 'summary.grd'):

    path = '%s/scores' % (os.environ['DEV_PATH'])
    folder = sname.split('-')[0]

    if not os.path.exists("%s/%s" % (path, folder )  ):
        os.makedirs( "%s/%s" % (path, folder ) )
        os.chown("%s/%s" % (path, folder ), 1000, 1000)

    with open('%s/%s/%s' % ( path, folder, sname) ,'w') as FSO:        
            FSO.writelines( scores )

    os.chown('%s/%s/%s' % (path, folder, sname ), 1000, 1000)
            

def f32(flt):
    if type(flt) is list or\
       type(flt) is np.ndarray:
        for i,arg in enumerate(flt):            
                flt[i] = f32(arg)

    elif type(flt) is float or \
         type(flt) is np.float64:
        flt = float(np.float32(flt)) 
    return flt


def get_sname():

    tme = time.localtime()
    dte = [tme.tm_year, tme.tm_mon, tme.tm_mday,\
           tme.tm_hour, tme.tm_min]

    for i,el in enumerate(dte):
        if float(el) < 10:
            dte[i] = '0%s' % str(el)
        else:
            dte[i] = str(el)

    sname = '%s_%s_%s-%s_%s.grd' % tuple( dte )

    return sname

def log_spacing( low_bound, high_bound, steps = 10 ):

    low = np.log10( low_bound )
    high = np.log10( high_bound )
    exp_list = np.linspace(low, high, steps)

    return np.power(10, exp_list)

def get_grid(x,y, x_key = 'x', y_key = 'y'):

    grid = np.meshgrid(x,y)

    lst =[]
    for i in range(grid[0].shape[0]):
        for j in range(grid[0].shape[1]):
            lst.append( {x_key: grid[0][i,j], y_key:grid[1][i,j] } )

    return lst

def get_new_args(args, layer, grid_element):

    lbl={'hex_mask':0, 
        'seed':1, 
        'nhid':2, 
        'nodes':3, 
        'lrinit':4, 
        'lrdecay':5, 
        'momentum_init':6, 
        'momentum_saturate':7, 
        'flag_reg':8}

    for l in layer.keys():
        args[ lbl[l] ] = layer[ l ]

    for g in grid_element.keys():
        args[ lbl[g] ] = grid_element[ g ]

    args = f32( args )

    return args


if __name__ == '__main__':
    main()