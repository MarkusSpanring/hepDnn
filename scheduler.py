###############################################################################
#                  Author: Markus Spanring HEPHY Vienna 2015                  #
###############################################################################
#
# Optimizes multible models defined by a fixed set of hyperparameters 
# with spearmint for a given number of iterations.
#
###############################################################################
import sys
import os
import time
import spearmint.main as spear
import cleanup

def main():
    '''Takes paramters stored in 'fixed_params' (one at a time)
       and optimizes with spearmint for 'max_jobs' iterations.
       'fixed_params' contains a list of dictionaries.
       Each dictionary contains the paramters to fix.
       When 'fixed_params' is created with 
       'create_mask_list' variable reduction is performed
       from all features down to one.
    '''

    top_mask = '0x0590'
    max_jobs = 25
    AMS_list = []

    while True:
        fixed_params = create_mask_list(fixed_mask = top_mask)
        #fixed_params = [{'nhid':1},{'nhid':2},{'nhid':3},\
        #                {'nhid':4},{'nhid':5},False]                      

        print 'Using: ', fixed_params

        smry_list = []
        for fixed_param in fixed_params:
            if not fixed_param:
                return AMS_list
            try:
                smry_list.append(spear.main( max_jobs = max_jobs,
                                             expt_dir = os.environ['DEV_PATH'],
                                             fixed_param = fixed_param))
                time.sleep(10)
                cleanup.main()
            except KeyboardInterrupt:
                sys.exit()

        top_mask = find_top_mask( smry_list )

def find_top_mask(summary_list):
    '''Reads all summaries in 'summary_list' and extracts the 
       best obtained result from all lists. 

       RETURN:
       Mask corresponding to best obtained result
    '''
    dev_path = os.environ['DEV_PATH']
    AMS = 0
    Mask = '0x000'
    search_pattern = ['#AMSrb_test: ', '#hex_mask: ']
    for summary in iter(summary_list):
        summary_folder = summary.split('-')[0]

        with open('%s/scores/%s/%s' % \
                 (dev_path, summary_folder, summary) , 'r') as FSO:
            strLines = FSO.readlines()

        for strLine in iter(strLines):
            if search_pattern[0] in strLine:
                tmpAMS = float(strLine.split(search_pattern[0])[1])

            if search_pattern[1] in strLine:
                tmpMask = strLine.split(search_pattern[1])[1]

        if tmpAMS > AMS:
            AMS = tmpAMS
            Mask = tmpMask

    return Mask



def create_mask_list(fixed_mask):
    '''Creates a list starting from fixed_mask 
        with masks where one feature was removed

        RETURN:
        list of dictionaries where each dict contains
        one mask.
    '''

    data_path = os.environ['PYLEARN2_DATA_PATH']

    intMask = int(fixed_mask,0)

    with open('%s/descr.dat' % data_path,'r') as FSO:
        nFeat_total = len(FSO.readline().replace('y,weight,','').split(','))

    if nFeat_total - bin(intMask).count('1') - 1 <= 0:
        return False

    for i in xrange(10):
        if 16**i > 2**nFeat_total:
            placeholder = 16**i
            break
    else:
        placeholder = 0
        return False

    new_mask_list = []

    for mask in reversed([2**i for i in xrange(nFeat_total)]):
        if (intMask & mask) != mask:
            hex_mask = hex(placeholder+intMask+mask).replace('0x1','0x')
            new_mask_list.append({'mask':hex_mask})

    return new_mask_list


if __name__ == '__main__':
    main()





