###############################################################################
#                  Author: Markus Spanring HEPHY Vienna 2015                  #
###############################################################################
#
# Provides functions to log spearmint optimization process. 
# 'write_summary': logs training process in file
# 'check_duplicate': reads summary file and checks if hyperparameters
#                    are unique
#
###############################################################################
import os
import numpy as np
import json

def mask_to_feat(hex_mask, mode = 'blocked'):
    '''Returns the features blocked/allowed (depending on 'mode')
       corresponding to 'hex_mask' 

       RETURN:
       mode = 'blocked': String with all blocked features
       mode = 'allowed': String with all allowed features
    '''


    path = os.environ['PYLEARN2_DATA_PATH']
    with open(path+"/descr.dat","r") as FSO:
        features_list = FSO.readline().replace("y,weight,","").split(',')    

    mask_list = reversed([2**i for i,v in enumerate(features_list)])

    blocked_features = []
    features = []

    for i,mask in enumerate(mask_list):
        if int(hex_mask,0)&mask == mask:
            blocked_features.append(features_list[i])
        else:
            features.append(features_list[i])

    if mode == 'blocked':
        return ' | '.join(blocked_features)
    elif mode == 'allowed':
        return ' | '.join(features)
    else:
        return ''

def write_summary(s_name = "summary.dat"):
    """
    Reads all files in the $DEV_PATH/ouput directory
    and extracts configuration and corresponding score
    Afterwards it writes a summary 's_name' in wich the top
    configuration and the complete trace of the 
    optimization process is written.

    [This function is ugly :( Rewrite recommended ]

    RETURN:
    Returns the maximal score from the optimization process.
    """


    dev_path = os.environ['DEV_PATH']
    filenames = [fl for fl in iter(sorted(os.listdir("%s/output" % dev_path)))\
                 if '.out' in fl]

    expt_file = '%s/config.json' % dev_path
    search_pattern = ['Completed successfully in ','Loss: ','score:']
    descr = ['AMSrb_test','AMS_test','AMSrb_train','AMS_train',\
             'ptype','hex_mask','seed','nhid','nodes','lrinit','lrdecay',\
             'momentum_init','momentum_saturate','momentum_final','batch_size']

    maxScore, loss, fltScore= [0.,0.,0.]
    strConf, ScoreSmry = ['','']
    read_conf = False
    s_name_dir = s_name.split('-')[0]

    if not os.path.exists("%s/scores/%s" % (dev_path, s_name_dir )  ):
        os.makedirs( "%s/scores/%s" % (dev_path, s_name_dir ) )
        os.chown("%s/scores/%s" % (dev_path, s_name_dir ), 1000, 1000)

    with open(expt_file, 'r') as f:
        options = json.load(f)
    expt_name = '#optimize with %s\n' % options['experiment-name']

    for filename in iter(filenames):
        with open('%s/output/%s' % (dev_path, filename) , 'r') as FSO_r:
            for strLine in iter(FSO_r.readlines()):

                if search_pattern[0] in strLine:
                    duration = strLine.split(' seconds')[0]\
                                      .replace(search_pattern[0],'')

                elif search_pattern[1] in strLine:
                    loss = strLine.split(search_pattern[1])[1]\
                                  .replace('\n','') 

                elif search_pattern[2] in strLine:

                    strScore = strLine.replace(search_pattern[2], '')
                    fltScore = float(strScore.split(',')[0])
                    maxScore = max(maxScore,fltScore)             
                    job = filename.replace('.out','')
                    ScoreSmry += ','.join([job, strScore.replace('\n',''),
                                      duration,loss,'\n'])               

                if max(maxScore, fltScore) ==  fltScore:
                    

                    if '-end-' in strLine:
                        read_conf = False

                    if read_conf:
                        strConf += strLine

                    if '-begin-' in strLine:
                        read_conf = True
                        strConf = ''

                        for i,element in enumerate(strScore.split(',')):

                            strConf += '%s: %s\n' % (descr[i], element)
                            if descr[i] is 'hex_mask':
                                strConf += 'allowed features: %s\n' %\
                                            mask_to_feat(hex_mask = element, 
                                                         mode = 'allowed')

                                strConf += 'blocked features: %s\n' %\
                                            mask_to_feat(hex_mask = element,
                                                         mode = 'blocked')


    if strConf != '':
        Conf = ''
        line ='-'*47
        for i, strLine in enumerate(strConf.split('\n')):

            if i == 0 or i == 4:
                Conf += '#%s\n' % line
            Conf += '#%s\n' % strLine


        with open('%s/scores/%s/%s' %\
            (dev_path, s_name_dir, s_name ),'w') as FSO_w:

            FSO_w.write(expt_name)
            FSO_w.write(Conf)
            FSO_w.write('#%strace%s\n' % (line,line))
            FSO_w.write(ScoreSmry)
        os.chown('%s/scores/%s/%s' %\
                 (dev_path, s_name_dir, s_name ), 1000, 1000)
        
    return maxScore

    

def check_duplicate(args, conf, s_name):
    """
    Checks if spearmint already tried the network
    with configuration 'args' by searching
    in the trace of the summary 's_name'.
    If there is already a entry, the return value is
    the corresponding AMS score and 'True'

    RETURN:
    list [DUPLICATE, score]

    DUPLICATE type=boolean
    score     type=float
    """
    dev_path = os.environ['DEV_PATH']
    s_path = '%s/scores/%s/%s' %\
             (dev_path, s_name.split('-')[0], s_name )

    DUPLICATE = False
    trace = ""
    score = 0
    modelname = '%s,%d,%d,%d,%0.14f,%0.14f,%0.6f,%d,%0.6f' % tuple(args[:-1])
    modelname = '%s,%s,%d' % (str( conf['ptype'] ),modelname,
                                    int( conf['batch_size']) )

    if not os.path.exists(s_path):
        open(s_path, 'w').close()
        os.chown(s_path, 1000, 1000)

    try:
        for entry in iter(np.loadtxt(s_path,dtype = np.str)):
            if modelname in entry:
                score = float(entry.split(',')[1])*-1
                DUPLICATE = True
    except:
        return [False,0.]     

    return [DUPLICATE,score]

def f32(flt):
    if type(flt) is list or\
       type(flt) is np.ndarray:
        for i,arg in enumerate(flt):            
                flt[i] = f32(arg)

    elif type(flt) is float or \
         type(flt) is np.float64:
         flt = float(np.float32(flt)) 
    return flt