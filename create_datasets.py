###############################################################################
#                  Author: Markus Spanring HEPHY Vienna 2015                  #
###############################################################################
# Recommended environment variables:
#   PYLEARN2_DATA_PATH= ../csv_data    -points to directory where 
#                                       csv-files are stored
#   DEV_PATH=../dev                    -points to directory where
#                                       python scripts are stored
#   RTFILES=../root_files              -points to directory where
#                                       root-files are stored
#
#
# Extracts datasets from a given set of root-files
# The data to read from a root-file is specified via
#   rt_name
#       tree_name:
#           BranchStruct{
#            datatype1 branch_name1
#            datatype2 branch_name2
#            }
#
# Possible file structure 1:
# In $RTFILES/folder_name:
#   test_ptype_sig_keyword1.root
#   test_ptype_sig_keyword2.root
#   test_ptype_bg_keyword1.root
#   test_ptype_bg_keyword2.root
#   train_ptype_sig_keyword1.root
#   train_ptype_sig_keyword2.root
#   train_ptype_bg_keyword1.root
#   train_ptype_bg_keyword2.root
#
# Possible file structure 2:
# In $RTFILES/folder_name:
#   signal1.root
#   signal2.root
#   background1.root
#   background2.root
###############################################################################
import os
import sys
import numpy as np
import shutil
import inspect
from ROOT import *



def main():
    '''
    Example of how to create datasets as csv-files
    '''
 
    strStruct = get_tree_struct(features = 'high')
    cd = Create_datasets(folder_name = '8TeV', tree_name = 'TauCheck', ptype = 'mu', strStruct = strStruct,ratio = 1.)
    cd.save_datasets_csv(ptypes = ['mu'])


class Create_datasets():
    '''
    folder_name             Folder in '$RTFILES'
    tree_name               Name of tree in root-file
    strStruct               Datastructure of tree
    ySig                    Predicted value for Signal
    yBg                     Predicted value for Background
    ratio                   Ratio is given by nBg/nSig
    forTesting              Percentage of events from full set
                            to use for training
    ptype                   particle type ('all' for the complete dataset)    
    sig_keywords            keywords in filename to specify as signal
    bg_keywords             keywords in filename to specify as background
    exclude                 keywords for files to exclude from datasets
    alt_DEV_PATH            alternative Paths 
    alt_RTFILES                 --//--
    alt_PYLEARN2_DATA_PATH      --//--

    '''
    def __init__(self, folder_name, tree_name, strStruct,
                 ySig = 1., 
                 yBg = 0.,
                 forTesting = 0.2,
                 ratio = 0.0001,
                 ptype = 'all', 
                 sig_keywords = ['vbfhiggs','ggfhiggs'], 
                 bg_keywords = ['dy1j','dy3j','dy2j','dy4j'], 
                 exclude = ['norecoil'],
                 alt_DEV_PATH = '',
                 alt_RTFILES = '',
                 alt_PYLEARN2_DATA_PATH = ''):
        try:
            if alt_DEV_PATH != '':
                dev_path = alt_DEV_PATH
            else:
                dev_path = os.environ['DNN_PATH']
        except:
            dev_path = '%s' % os.path.expanduser('~')

        try:
            if alt_PYLEARN2_DATA_PATH != '':
                self.data_path = alt_PYLEARN2_DATA_PATH
            else:
                self.data_path = os.environ['PYLEARN2_DATA_PATH']
        except:
            self.data_path = '%s/csv_data' % os.path.expanduser('~')

        try:
            if alt_RTFILES != '':
                self.rt_path = alt_RTFILES
            else:
                self.rt_path = os.environ['RTFILES']
        except:
            print 'No idea where to find root files'
            sys.exit() 


        gROOT.ProcessLine(strStruct)

        self.folder_name = folder_name
        self.tree_name = tree_name
        self.ySig = ySig
        self.yBg = yBg
        self.forTesting = forTesting
        
        if ratio < 0.0001:
            ratio = 0.0001
        self.ratio = ratio
        self.ptype = ptype        
        self.sig_keywords = sig_keywords
        self.bg_keywords = bg_keywords
        self.exclude = exclude
        self.descr = ''
        args = locals()
        self.datasets = ''
        for key in args:
            if key != 'ptype':
                self.datasets += '%s: %s\n' % ( key, args[key] )
                self.datasets = self.datasets.replace('{','\n').replace(';','\n')

        blnSig = False
        blnBg = False

        for el in self.sig_keywords:     
            if 'signal' in el: blnSig = True

        for el in self.bg_keywords:     
            if 'background' in el: blnBg = True

        if not blnSig: self.sig_keywords.append('signal')
        if not blnBg: self.bg_keywords.append('background')

    def set_parameter(self, folder_name = None, 
                      tree_name = None, 
                      ySig = None, 
                      yBg = None, 
                      forTesting = None, 
                      ptype = None, 
                      sig_keywords = None, 
                      bg_keywords = None, 
                      exclude = None ):

        '''
        Allows to set or reset an arbitrary number of parameters
        '''
        if folder_name != None:
            self.folder_name = folder_name

        if tree_name != None:
            self.tree_name = tree_name

        if ySig != None: 
            self.ySig = ySig

        if yBg != None:
            self.yBg = yBg

        if forTesting != None:
            self.forTesting = forTesting

        if ptype != None:
            self.ptype = ptype

        if sig_keywords != None:        
            self.sig_keywords = sig_keywords

        if bg_keywords != None:
            self.bg_keywords = bg_keywords

        if exclude != None:
            self.exclude = exclude



    def keywords(self, rt_name):
        '''
        Checks if sig_keywords, bg_keywords or exclude keywords
        are in the root-filname 'rt_name' and returns the corresponding 
        predicted y-value

        RETURN:
        for sig_keyword in rt_name -> ySig    type=float
        for bg_keyword in rt_name -> yBg      type=float
        for excluded in rt_name -> 'excluded' type=string
        '''

        blnSig = False
        blnBg = False
        for keyword in self.sig_keywords:
            if keyword in rt_name:
                blnSig = True

        for keyword in self.bg_keywords:
            if keyword in rt_name:
                blnBg = True

        for keyword in self.exclude:
            if keyword in rt_name:
                return 'excluded'

        if blnSig: return self.ySig
        elif blnBg: return self.yBg
        else: return 'excluded'


    def Get_Entries(self,rt_name):
        '''
        Returns the number of entries in root-file 'rt_name'
        '''

        descr = ''

        inputfile = "%s/%s/%s" % (self.rt_path,self.folder_name ,rt_name)

        branch_available = {}
        file0 = TFile(inputfile,"read")
        tree = file0.Get( self.tree_name )
        nEntries = tree.GetEntries()
        file0.Close()
        return nEntries




    def ReadRoot(self,rt_name, splitFactor = 1.):
        '''
        Reads all branches specified in 'strStruct' from 
        tree 'tree_name' in the root-file 'rt_name'
        and returns the dataset in combination with a 
        string describing the structure of the dataset.
        The first column contains the predicted value 
        'ySig' or 'yBg'

        RETURN:
        list [dataset, description]:

        dataset type=list           dataset['ySig/yBg', branch1_value1, branch2_value1]
                                            'ySig/yBg', branch1_value2, branch2_value2]

        description type=string     descr =   y,      branch1_name,   branch2_name   

        '''

        lstAll = []
        descr = ''

        inputfile = "%s/%s/%s" % (self.rt_path, self.folder_name , rt_name)

        branch_available = {}
        file0 = TFile(inputfile,"read")
        tree = file0.Get( self.tree_name )
        ProcLine = BranchStruct_t()
        Y = self.keywords(rt_name = rt_name)
        if Y == 'excluded': return [np.array([]),'']

        for branch in inspect.getmembers(ProcLine):

          if not '__' in str(branch):

            if tree.GetBranchStatus( str( branch[0] ) ):
                branch_available[ str( branch[0] ) ] = True

                tree.SetBranchAddress( str( branch[0] ),AddressOf( ProcLine,str( branch[0] ) ) )

            else:
                branch_available[ str( branch[0] ) ] = False


        for i in xrange(tree.GetEntries()):
            tree.GetEntry(i)

            weight = 1.
            if branch_available['weight']:            
                weight *= ProcLine.weight            

            if branch_available['lumiWeight']: 
                weight *= ProcLine.lumiWeight
                
            if branch_available['splitFactor']: 
                weight *= ProcLine.splitFactor

            else:
                weight *= splitFactor        

            tmp = [Y,weight]
            descr = 'y,weight'


            for branch in inspect.getmembers(ProcLine):
                if not '__' in str(branch):
                    try:
                        if ( str(branch[0] ) == 'weight' ) or\
                           ( str(branch[0] ) == 'lumiWeight' ) or\
                           ( str(branch[0] ) == 'splitFactor' ) or\
                           ( str(branch[0] ) == 'lep_isM' ) or\
                           ( str(branch[0] ) == 'lep_isE' ): pass

                        elif branch_available[ str( branch[0] ) ]:
                            tmp.append( float( branch[1] ) )
                            descr += ',%s' % str( branch[0] )
                    except KeyError: pass


            if self.ptype == 'mu':
                if branch_available['lep_isM']: 
                    if ProcLine.lep_isM == 1:
                        lstAll.append(tmp)

                elif  (self.ptype in rt_name):
                    lstAll.append(tmp)


            elif self.ptype == 'el':
                if branch_available['lep_isE']: 
                    if ProcLine.lep_isE == 1:
                        lstAll.append(tmp)

                elif (self.ptype in rt_name):
                    lstAll.append(tmp)

            else:
                lstAll.append(tmp)


        file0.Close()
        if descr != '':
            self.descr += descr + '\n'

        return [np.array(lstAll), descr]

    def get_data(self,signal_type):
        '''
        Is used to read data from file structure 2
        Reads all files in '$RTFILES/folder_name'
        which belong to 'signal_type'. The data is then split into
        a Test, Train and Monitor Dataset. 

        RETURN:
        type=list [Test_set, Training_set, description]

        type=list    Test_set = forTesting * Complete_set
        type=list    Training_set = (1-forTesting) * Complete_set
        type=string  description
        '''

        try:
            filenames = os.listdir('%s/%s' % ( self.rt_path, self.folder_name ) )
        except:
            print 'Error while getting files in %s' % self.folder_name
            raise IOError

        if self.forTesting > 0:
            splitFactor = 1.0/self.forTesting
        else: splitFactor = 1

        first = True
        for filename in filenames:
            if signal_type in filename:
                lstAll, descr = self.ReadRoot(rt_name = filename,
                                         splitFactor = splitFactor)

                nTest = int( lstAll.shape[0] * self.forTesting )
                if first and (lstAll != []):
                    

                    lstTest = lstAll[0:nTest]
                    lstTrain = np.delete(lstAll , [i for i in xrange(nTest)] ,0)
                    first = False
                else:
                    if lstAll != []:

                        lstTest = np.vstack( [ lstTest , lstAll[0:nTest] ] )
                        lstTrain = np.vstack( [ lstTest , np.delete(lstAll , [i for i in xrange(nTest)] ,0) ] )


        return [lstTest, lstTrain,descr]



    def compose_data(self):
        sigTest, sigTrain, descr = self.get_data('signal')
        bgTest, bgTrain, descr = self.get_data('background')

        nBgtest = float( bgTest.shape[0] )
        nSigtest = float( sigTest.shape[0] )

        sigTest = sigTest[0:nBgtest]
        sigTest[:,1] = sigTest[:,1]*( nSigtest/nBgtest )


        sigTrain[:,1] = sigTrain[:,1]/(1 - self.forTesting)
        bgTrain[:,1] = bgTrain[:,1]/(1 - self.forTesting)


        rng = np.random.RandomState(42)
        bgInd = np.arange(bgTrain.shape[0])
        sigInd = np.arange(sigTrain.shape[0])
        rng.shuffle(bgInd)
        rng.shuffle(sigInd)
        bgTrain = bgTrain[bgInd, :]
        sigTrain = sigTrain[sigInd, :]


        lstTest = np.vstack( [ sigTest, bgTest ] )
        nTest = lstTest.shape[0]

        nSig = float( sigTrain.shape[0] )
        nBg= float( bgTrain.shape[0] )


        ratio = nBg/nSig


        if (ratio < self.ratio) and self.ratio <= 1.0:
            sigTrain = sigTrain[0: int(nBg/self.ratio)]
            splFact = nSig /  int(nBg/self.ratio)
            sigTrain[:,1] = sigTrain[:,1]*splFact
            lstTrain = np.vstack([ sigTrain, bgTrain ] )            

        elif (ratio > self.ratio) and self.ratio <= 1.0:
            lstTrain = np.vstack([ sigTrain, bgTrain ] ) #


        elif (ratio < self.ratio) and self.ratio > 1.0:
            sigTrain = sigTrain[0: int(nBg/self.ratio)]
            splFact = nSig /  int(nBg/self.ratio)
            sigTrain[:,1] = sigTrain[:,1]*splFact
            lstTrain = np.vstack([ sigTrain, bgTrain ] )

        elif (ratio > self.ratio) and self.ratio > 1.0:
            lstTrain = np.vstack([ sigTrain, bgTrain ] )#

        else:
            lstTrain = np.vstack([ sigTrain, bgTrain ] )#

        print nSig, sigTrain.shape[0], nBg, bgTrain.shape[0]


        indices = np.arange(lstTrain.shape[0])
        rng.shuffle(indices)
        lstTrain = lstTrain[indices, :]



        lstMonitor = np.vstack([ sigTrain[0:int(sigTrain.shape[0]*self.forTesting)],
                                  bgTrain[0:int(bgTrain.shape[0]*self.forTesting)] ] ) 


        return [lstTest,lstTrain,lstMonitor,descr]

    def merge_files(self):
        '''
        Used to read data from file structure 1
        Reads all files in '$RTFILES/folder_name'
        which belong to 'test' and 'train'. Creates 
        from 'test'-files a Test_set and from 
        'train'-files a Training_set.
        From the Training_set the same number 
        of events as in the Test_set is used to create
        a Monitor_set

        RETURN:
        type=list [Test_set, Training_set, Monitor_set, description]

        type=list    Test_set
        type=list    Training_set        
        type=list    Monitor_set = same number as test events from Training_set
        typestring   description
        '''

        try:
            filenames = os.listdir('%s/%s' % ( self.rt_path, self.folder_name ) )
        except:
            print 'Error while getting files in %s' % self.folder_name

        first_test = True
        first_train = True
        for filename in filenames:


            if 'test' in filename:
                if ( self.ptype in filename ) or (self.ptype == 'all') :

                    lstTest, descr = self.ReadRoot(rt_name = filename)

                    if first_test and ( lstTest != [] ):
                        lstTestAll = lstTest
                        first_test = False
                    else:
                        if lstTest != []: 
                            lstTestAll = np.vstack([ lstTestAll ,lstTest ] )

            elif 'train' in filename:
                if ( self.ptype in filename ) or (self.ptype == 'all'):

                    lstTrain, descr = self.ReadRoot(rt_name = filename)

                    if first_train and ( lstTrain != [] ):
                        lstTrainAll = lstTrain
                        first_train = False
                    else:
                        if lstTrain != []: 
                            lstTrainAll = np.vstack([ lstTrainAll ,lstTrain ] )

        rng = np.random.RandomState(42)
        indices = np.arange(lstTrainAll.shape[0])
        lstTrainAll = lstTrainAll[indices, :]
                       

        lstMonitorAll = lstTrainAll[ 0 : lstTestAll.shape[0] ]

        return [lstTestAll,lstTrainAll,lstMonitorAll,descr]

    def get_datasets_full(self):
        '''
        Returns from '$RTFILES/folder_name' a 
        Training, Test and Monitor dataset which
        contain  labeled signal as well as labeld  background 
        data. 

        RETURN:
        type=list [Test_set, Training_set, Monitor_set, description]

        type=list    Test_set
        type=list    Training_set        
        type=list    Monitor_set = same number as test events from Training_set
        typestring   description
        '''

        
        try:
            filenames = os.listdir('%s/%s' % ( self.rt_path, self.folder_name ) )
        except:
            print 'Error while getting files in %s' % self.folder_name

        blnTest = False
        blnTrain = False
        blnSig = False
        blnBg = False


        for filename in filenames:
            if 'test' in filename: blnTest = True
            if 'train' in filename: blnTrain = True
            if 'signal' in filename: blnSig = True
            if 'background' in filename: blnBg = True

        if blnTest and blnTrain:
            lstTestAll, lstTrainAll, lstMonitorAll, descr =\
            self.merge_files()

        elif blnSig and blnBg:

            lstTestAll, lstTrainAll, lstMonitorAll, descr =\
            self.compose_data()


        self.test_descriptions()
        return [lstTestAll, lstTrainAll, lstMonitorAll, descr]

    def test_descriptions(self):
        first = True
        equal = True
        for strLine in self.descr.split('\n'):
            if first:
                tmpDescr = strLine
                first = False
            elif strLine != '':
                if tmpDescr != strLine:
                    equal = False

        if not equal:
            raise IOError('Descriptions do not match!!!')

    def save_datasets_csv(self, ptypes = ''):
        '''
        Writes all datasets coresponding to 'ptypes'
        (ptypes can be a list [ptype1,ptype2,..] or a single ptype
        ptypes = ptype) and writes them to '$PYLEARN2_DATA_PATH'
        including a description
        '''       

    
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        else:
            shutil.rmtree(self.data_path)
            os.makedirs(self.data_path)

        if ptypes == '': ptypes = [self.ptype]

        #try:
        for ptype in ptypes:
            self.set_parameter(ptype=ptype)

            lstTest, lstTrain, lstMonitor, descr =\
            self.get_datasets_full()
            
            np.savetxt('%s/train_%s.dat' % ( self.data_path, self.ptype ), lstTrain, delimiter = ',', fmt = "%.14f")
            np.savetxt('%s/monitor_%s.dat' % ( self.data_path, self.ptype ), lstMonitor, delimiter = ',', fmt = "%.14f")
            np.savetxt('%s/test_%s.dat' % ( self.data_path, self.ptype ), lstTest, delimiter = ',', fmt = "%.14f")

        with open("%s/descr.dat" % self.data_path,"w") as FSO:
            FSO.write(descr)

        with open("%s/datasets.dat" % self.data_path,"w") as FSO:
            self.datasets += 'ptypes: %s' % ptypes
            FSO.write(self.datasets)
            
        #     print 'writing to %s ***successfull***' % self.data_path
        # except:
        #     print 'writing to %s ***NOT*** successfull' % self.data_path

def get_tree_struct(features = 'high'):

    if features == 'high':
            strStruct = \
            "struct BranchStruct_t {\
            Float_t weight;\
            Float_t lumiWeight;\
            Float_t splitFactor;\
            Float_t svfit_mass;\
            Float_t dr_leptau;\
            Float_t jdeta;\
            Float_t mjj;\
            Float_t jeta1eta2;\
            Float_t pt_tot;\
            Float_t met_centrality;\
            Float_t mt_1;\
            Float_t lep_etacentrality;\
            Float_t pt_sum;\
            Float_t sphericity;\
            Float_t mvis;\
            Float_t svfit_pt;\
            Float_t lep_isM;\
            Float_t lep_isE;\
            }"

    elif features == 'low':
            strStruct = \
            "struct BranchStruct_t {\
            Float_t weight;\
            Float_t lumiWeight;\
            Float_t splitFactor;\
            Float_t tau_pt;\
            Float_t tau_eta;\
            Float_t tau_phi;\
            Float_t lep_pt;\
            Float_t lep_eta;\
            Float_t lep_phi;\
            Float_t met_pt;\
            Float_t met_phi;\
            Float_t jet1_pt;\
            Float_t jet1_eta;\
            Float_t jet1_phi;\
            Float_t jet2_pt;\
            Float_t jet2_eta;\
            Float_t jet2_phi;\
            Float_t jet3_pt;\
            Float_t jet3_eta;\
            Float_t jet3_phi;\
            Float_t jet4_pt;\
            Float_t jet4_eta;\
            Float_t jet4_phi;\
            Float_t lep_isM;\
            Float_t lep_isE;\
            }"

    else:
            strStruct = \
            "struct BranchStruct_t {\
            Float_t weight;\
            Float_t lumiWeight;\
            Float_t splitFactor;\
            Float_t svfit_mass;\
            Float_t dr_leptau;\
            Float_t jdeta;\
            Float_t mjj;\
            Float_t jeta1eta2;\
            Float_t pt_tot;\
            Float_t met_centrality;\
            Float_t mt_1;\
            Float_t lep_etacentrality;\
            Float_t pt_sum;\
            Float_t sphericity;\
            Float_t mvis;\
            Float_t svfit_pt;\
            Float_t tau_pt;\
            Float_t tau_eta;\
            Float_t tau_phi;\
            Float_t lep_pt;\
            Float_t lep_eta;\
            Float_t lep_phi;\
            Float_t met_pt;\
            Float_t met_phi;\
            Float_t jet1_pt;\
            Float_t jet1_eta;\
            Float_t jet1_phi;\
            Float_t jet2_pt;\
            Float_t jet2_eta;\
            Float_t jet2_phi;\
            Float_t jet3_pt;\
            Float_t jet3_eta;\
            Float_t jet3_phi;\
            Float_t jet4_pt;\
            Float_t jet4_eta;\
            Float_t jet4_phi;\
            Float_t lep_isM;\
            Float_t lep_isE;\
            }"

    return strStruct


            




if __name__ == '__main__':
    main()