#TAMS
#Author: Martin Flechl, 2014
#calculate average median significance of two histograms
###############################################################################
#                  Author: Markus Spanring HEPHY Vienna 2015                  #
###############################################################################


import os
import argparse
import numpy as np
import theano
import sys
import pylearn2.datasets.physics
import pickle as pkl
from pyAMS import TAMS
try:import ROOT
except ImportError:
    sys.path.insert(0, "/usr/lib64/python2.6/site-packages")
    import ROOT


class ModelAMS():

    def __init__(self, modelname, ptype = "all", seed=42, flag_reg = 0x3):
        self.modelname = modelname
        self.rt_name = modelname.replace('pkl','root')
        self.ptype = ptype
        self.seed = seed
        self.flag_reg = flag_reg
        self.score = {"test_rb":0, "test":0, "train_rb":0, "train":0}

        if flag_reg&0x1 == 0x1:
            self.contains_weight = True
        else: self.contains_weight = False

        if flag_reg&0x2 == 0x2:
            self.contains_y = True
        else: self.contains_y = False

        if self.contains_weight and self.contains_y:
            self.his_descr = "w_out"
        elif self.contains_y:
            self.his_descr = "out"
        else:
            self.his_descr = "sig_p"

    def fprop(self, model, X):
        '''
        Propagate the data through network and return
        activation
        '''
        
        
        X_theano = model.get_input_space().make_theano_batch()
        Y_theano = model.fprop(X_theano)
        f = theano.function( [X_theano], Y_theano )

        return f(X)


    def model_yhat(self):
        """
        Uses model: modelname an propagates 'test' and 'train' dataset through
        network. 

        RETURN:
        [Yhat_test, Yhat_train]

        """

        dev_path = os.environ['DEV_PATH']

        print "\npropagating through: "+self.modelname
        try:
            model = pkl.load(open("%s/model/%s"% (dev_path,self.modelname),'r'))
        except:
            return [None,None]
                
        hex_mask = self.modelname.split("_")[1]
        dataset_test = pylearn2.datasets.physics.PHYSICS(which_set="test",
                                                         ptype = self.ptype,
                                                         seed = self.seed,
                                                         flag_reg = self.flag_reg,
                                                         hex_mask = hex_mask)

        dataset_train = pylearn2.datasets.physics.PHYSICS(which_set="monitor",
                                                          ptype = self.ptype,
                                                          seed = self.seed,
                                                          flag_reg = self.flag_reg,
                                                          hex_mask = hex_mask)

        Yhat_test = self.fprop(model = model,
                               X = dataset_test.X)

        Y_test = np.hstack((dataset_test.y,Yhat_test))
        if self.contains_weight:
            Y_test = np.hstack( ( Y_test,dataset_test.event_weights ) )

        Yhat_train = self.fprop(model = model,
                                X = dataset_train.X)

        Y_train = np.hstack( (dataset_train.y,Yhat_train) )
        if self.contains_weight:
            Y_train = np.hstack( ( Y_train,dataset_train.event_weights ) )

        return [Y_test,Y_train]

    def data_to_hist(self, data):

            """
            Writes given data to TH1F Histogram

            RETURN:

            if data[0] is None '-1' will be returned

            If Histogram is filled succesfully '1' will be returned
            """

            dev_path = os.environ['DEV_PATH']

            if not os.path.exists("%s/hist" % dev_path):
                os.makedirs("%s/hist" % dev_path)

            if data[0] is None:
                return -1

            f = ROOT.TFile("%s/hist/%s" % (dev_path, self.rt_name) ,"RECREATE")

            dset = ["test","train"]
            hBg = []
            hSig = []

            for i,entry in enumerate(data):

                bg_string = "%s_bg_%s" % ( self.his_descr, dset[i] )
                hBg.append( ROOT.TH1F(bg_string,bg_string, 1000,0.0,1.0) )
                if self.contains_y:
                    sig_string = "%s_sig_%s" % ( self.his_descr, dset[i] )
                    hSig.append(ROOT.TH1F(sig_string,sig_string, 1000,0.0,1.0))


                for element in entry[:,0:3]:
                    if self.contains_weight and self.contains_y:
                        if element[0] <= 0.0:
                            hBg[i].Fill(element[1],element[2])
                        if element[0] >= 1.0:
                            hSig[i].Fill(element[1],element[2])

                    elif self.contains_y:
                        if element[0] <= 0.0:
                            hBg[i].Fill(element[1])
                        if element[0] >= 1.0:
                            hSig[i].Fill(element[1])
                    else:
                        hBg[i].Fill(element[1])

            f.Write()
            f.Close()
            return 1

    def ams_hist(self):

            """
            Calculates the AMS value from a given Histogram

            RETURN:
            String containing the AMS value from 'test' and 'train'
            set and the corresponding model parameters.
            """

            dev_path = os.environ['DEV_PATH']


            strAMS = ""



            if os.path.isfile("%s/hist/%s" % (dev_path, self.rt_name)):

                try:
                    f0 = ROOT.TFile("%s/hist/%s" % (dev_path, self.rt_name))
                    hSig_test = f0.Get(self.his_descr+"_sig_test")
                    hSig_train = f0.Get(self.his_descr+"_sig_train")

                    hBg_test = f0.Get(self.his_descr+"_bg_test")            
                    hBg_train = f0.Get(self.his_descr+"_bg_train")

                    tams = TAMS(hSig = hSig_test, hBg = hBg_test)

                    tams.br = 1
                    self.score["test"] = tams.ams_syst_stat(0)
                    tams.br = 0.001
                    self.score["test_reg"] = tams.ams_syst_stat(0)
                    tams.rebinEqui()
                    tams.br = 1
                    self.score["test_rb"] = tams.ams_syst_stat(0)
                    tams.br = 0.001
                    self.score["test_rb_reg"] = tams.ams_syst_stat(0)


                    tams.seth(hSig = hSig_train, hBg = hBg_train)
                    tams.br = 1
                    self.score["train"] = tams.ams_syst_stat(0)
                    tams.br = 0.001
                    self.score["train_reg"] = tams.ams_syst_stat(0)
                    tams.rebinEqui()
                    tams.br = 1
                    self.score["train_rb"] = tams.ams_syst_stat(0)
                    tams.br = 0.001
                    self.score["train_rb_reg"] = tams.ams_syst_stat(0)

                    f0.Close()
                except:
                    pass

                strAMS += "%.5f,%.5f," % (self.score["test_rb"],self.score["test"])
                strAMS += "%.5f,%.5f," % (self.score["train_rb"],self.score["train"])


                strAMS += self.rt_name.replace(".root","").replace("_",",")

                return strAMS

    def ams_model(self):
        """
        Calculates the AMS value from a given model
        """

        if self.data_to_hist( data = self.model_yhat() ) != -1:

            return self.ams_hist()
        else:
            return ",".join(["0","0","0","0",self.modelname.replace(".root","").replace("_",",")])


    def plot_hist(self):


            dev_path = os.environ['DEV_PATH']

            if os.path.isfile("%s/hist/%s" % (dev_path, self.rt_name )):

                if self.score['test'] == 0: self.ams_hist()

                f0 = ROOT.TFile("%s/hist/%s" % (dev_path, self.rt_name ) )
                hSig_test = f0.Get(self.his_descr+"_sig_test")
                hBg_test = f0.Get(self.his_descr+"_bg_test")            


                tams = TAMS(hSig = hSig_test, hBg = hBg_test)

                fname = self.modelname.replace('.pkl','')
                if '.root' in modelname:
                    fname = self.modelname.replace('.root','')

                #tams.savePlot(fname= '%s_raw.png' % fname)
                tams.rebinEqui()
                tams.savePlot(fname= '%s_rebin.png' % fname )
                f0.Close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-plot', help='Plot Histogram from model', type=str,metavar='MODEL', nargs='+')
    parser.add_argument('-ptype', help='Particle type to propagete through model', type=str,metavar='PTYPE')
    parser.add_argument('-ams', help='Calculate AMS from model', type=str,metavar='MODEL', nargs='+')
    parser.set_defaults(ptype='mu')

    args = vars( parser.parse_args() )

    if args['plot'] != None or  args['ams'] != None:

        if args['plot'] != None:
            if len(args['plot']) > 1:
                modelname = args['plot'][0]
                ptype = args['plot'][1]
            else:
                modelname = args['plot'][0]
                ptype = args['ptype']

            AMS = ModelAMS(modelname = modelname, ptype = ptype)
            AMS.plot_hist()

        if args['ams'] != None:
            if len(args['ams']) > 1:
                modelname = args['ams'][0]
                ptype = args['ams'][1]
            else:
                modelname = args['ams'][0]
                ptype = args['ptype']

            AMS = ModelAMS(modelname = modelname, ptype = ptype)

            if '.pkl' in modelname: source = 'model'
            else: source = 'hist' 
            
            if source == 'hist':
                    print AMS.ams_hist()
                    AMS.plot_hist()
            else:
                print AMS.ams_model()













