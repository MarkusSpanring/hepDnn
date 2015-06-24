###############################################################################
#                  Author: Markus Spanring HEPHY Vienna 2015                  #
###############################################################################
#
# Creates pylearn2 dataset from csv-files
#
###############################################################################

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from pylearn2.datasets import dense_design_matrix


def get_file_path(path,file_type,ptype): 


        filenames = os.listdir(path)
        if file_type is 'valid':
            file_type = 'train'

        for filename in filenames:
            if ptype in filename:
                if file_type in filename:

                    return '%s/%s'%(path,filename)

        return None


def get_n_samples(path,ptype):

        if get_file_path(path,'train',ptype) is not None:


            with open(get_file_path(path,'train',ptype),'r') as FSO:
                train_Events = len(FSO.readlines())        

            return [int(train_Events*0.95),int(train_Events*0.05)]
        else:
            raise [0,0]


class DATASET(dense_design_matrix.DenseDesignMatrix):
    """
    Creates train, test and validation datasets and returns them
    as DenseDesignMatrix.
    """
    def __init__(self, 
                 which_set,
                 ptype = "mu",
                 flag_reg=0x3,
                 hex_mask="0x0000",
                 seed=None,          #Randomize data order if seed is not None
                 start=0, 
                 stop=np.inf):

        dev_path = os.environ['DEV_PATH']

        if flag_reg&0x1 == 0x1:
            contains_weight = True
        else: contains_weight = False

        if flag_reg&0x2 == 0x2:
            contains_y = True
        else: contains_y = False

        if flag_reg&0x8 == 0x8:
            STDIZE = True
        else:
            STDIZE = False

        path = os.environ['PYLEARN2_DATA_PATH']
        shift=0


        if get_file_path(path, which_set, ptype) is not None:
            inputfile = get_file_path(path, which_set, ptype)

        else:
            raise IOError('File %s not found: %s|%s|%s' % (get_file_path(path, which_set, ptype),path, which_set, ptype ) )

        ntrain, nvalid = get_n_samples(path,ptype)


        print "using "+inputfile 
        
        X = np.loadtxt(inputfile, dtype='float32', delimiter=',')

        if contains_y:
            y = X[:,shift].reshape((-1,1))
            y = np.array(y, dtype='float32')
            shift += 1

        if contains_weight:
            event_weights = X[:,shift].reshape((-1,1))
            self.event_weights = np.array(event_weights, dtype='float32')
            shift += 1 

        X = X[:,shift:]
        X = self.apply_mask(data=X, hex_mask=hex_mask)        
        X = np.array(X, dtype='float32')
 


        print 'Data loaded: set %s.' % (which_set)




        # Randomize data order.
        if seed is not None:

            rng = np.random.RandomState(seed)
            indices = np.arange(X.shape[0])
            rng.shuffle(indices)
            X = X[indices, :]

            if contains_weight:
                self.event_weights = self.event_weights[indices, :]

            if contains_y:
                y = y[indices, :]

        if which_set == 'train':
            X = X[0:ntrain, :]

            if contains_weight:
                self.event_weights = self.event_weights[0:ntrain, :]
            if contains_y:
                y = y[0:ntrain, :]

        
        elif which_set == 'valid':

            X = X[ntrain:ntrain+nvalid, :]

            if contains_weight:
                self.event_weights = self.event_weights[ntrain:ntrain+nvalid, :]
            if contains_y:
                y = y[ntrain:ntrain+nvalid, :]


        # Limit number of samples.
        stop = min(stop, X.shape[0])
        X = X[start:stop, :]

        if contains_weight:
            self.event_weights = self.event_weights[start:stop, :]
        if contains_y:
            y = y[start:stop, :]

        if STDIZE:
            X = self.standardize(X, mode = 'sck')

        # Initialize the superclass. DenseDesignMatrix
        if contains_y:
            super(PHYSICS,self).__init__(X=X, y=y)
        else:
            super(PHYSICS,self).__init__(X=X)

    def apply_mask(self,data,hex_mask = "0x0000"):

        path = os.environ['PYLEARN2_DATA_PATH']
        with open(path+"/descr.dat","r") as FSO:
            features = FSO.readline().replace("y,weight,","").split(",")

        mask_list = reversed([2**i for i,v in enumerate(features)])
        mask_index = []

        for i,mask in enumerate(mask_list):
            if int(hex_mask,0)&mask == mask:
                mask_index.append(i)

        data=np.delete(data,mask_index,1)

        return data
        
    def standardize(self, X, mode = 'hardcoded'):
        """
        Standardize each feature:
        1) If data contains negative values, we assume its either
           normally or uniformly distributed, center, and standardize.
        2) elseif data has large values, we set mean to 1.
        """
        if mode == 'hardcoded':
            for j in range(X.shape[1]):
                vec = X[:, j]
                if np.min(vec) < 0:
                    # Assume data is Gaussian or uniform -- center and standardize.
                    vec = vec - np.mean(vec)
                    vec = vec / (np.std(vec))
                elif np.max(vec) > 1.0:
                    # Assume data is exponential -- just set mean to 1.
                    vec = vec / np.mean(vec)
                X[:,j] = vec
            return X

        else:
            scaler = StandardScaler().fit(X)
            return scaler.transform(X)






