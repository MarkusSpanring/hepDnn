import numpy as np 
import os
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate



class Plot():
    def __init__(self, s_name, x_param = 'nhid', y_param='nodes' ):
        self.dev_path = os.environ['DNN_PATH']
        self.LOGY = False
        self.LOGX = False

        if not os.path.exists("%s/figure" % self.dev_path):
            os.makedirs("%s/figure" % self.dev_path)

        if not os.path.exists("%s/figure/plot" % self.dev_path):
            os.makedirs("%s/figure/plot" % self.dev_path)

        self.param = {'seed':5,'nhid':6,'nodes':7,'lrinit':8,'lrdecay':9,\
                      'momentum_init':10,'momentum_saturate':11,'momentum_final':12,'batch_size':13,'loss':15}

        self.lbl = {'mask':'Mask', 'seed':5,'nhid':'Number of Hidden Layers','nodes':'Number of Nodes','lrinit':'Learnrate Initialization',\
        'lrdecay':'Learnrate Decay Factor','momentum_init':'Momentum Initialization',\
        'momentum_saturate':'Number of Epochs after Momentum saturates','momentum_final':'Final Momentum','batch_size':'Batch Size',\
        'loss': 'Loss'}

        try:
            self.npTrace = np.loadtxt('%s/scores/%s' % ( self.dev_path, s_name) ,delimiter=',',dtype='S10')
        except IOError:
            print 'File not found.'
            return

        self.npMask = self.npTrace[:,6]
        self.npMask = [int(el, 16) for el in self.npMask]
        print np.delete(self.npTrace,[5,6],1)[0]
        self.npTrace = np.array( np.delete(self.npTrace,[5,6,18],1),dtype = np.float32)

        self.x_param = ''
        self.y_param = ''

        self.z = self.npTrace[:,1]

    def set_data(self,x_param = 'nhid', y_param = 'nodes'):
        self.x_param = x_param
        self.y_param = y_param

        try:
            self.x = self.npTrace[:,self.param[x_param] ]
        except KeyError:
            if x_param == 'mask':
                self.x = self.npMask
            else:
                print 'unknown parameter: %s' % x_param
        try:
            self.y = self.npTrace[:,self.param[y_param] ]
        except KeyError:
            if y_param == 'mask':
                self.y = self.npMask
            else:
                print 'unknown parameter: %s' % y_param
                return

        if self.x.max()/self.x.min() > 100 or x_param == 'lrdecay':
            if x_param == 'lrdecay':
                self.x[:] -= 1
            self.x = np.log10(self.x)
            self.LOGX = True



        if self.y.max()/self.y.min() > 100 or y_param == 'lrdecay':
            if y_param == 'lrdecay':
                self.y[:] -= 1
            self.y = np.log10(self.y)
            self.LOGY = True



    def Plot_3D(self, x_param = 'nhid', y_param = 'nodes'):

        if not self.x_param == x_param and not self.y_param == y_param:

            self.set_data(x_param, y_param)

        fig = plt.figure()
        fig.canvas.set_window_title('Ploting %s over %s' % (x_param,y_param))
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(self.x, self.y, self.z, cmap=cm.jet, linewidth=0.2)
        ax.set_xlabel(self.lbl[x_param])
        ax.set_ylabel(self.lbl[y_param])
        ax.set_zlabel('AMS')
        plt.show()

    def Plot_Trace(self, x_param = 'AMS', show_fig = True):
        max_list = np.array([0,0])
        max_val = 0.

        PLOT_LOG = False

        if self.param.has_key( x_param ):
            y = self.npTrace[:,self.param[x_param]]
            mx = max(y)
            mn = min(y)
            if mx/mn > 1000 or x_param == 'lrdecay':
                if x_param == 'lrdecay':
                    tmp = []
                    for el in y:
                        tmp.append(el-1.0)
                    y = tmp
                    mn -= 1
                    mx -= 1 
                PLOT_LOG = True

        else:
            y = self.npTrace[:,1]

        if len(self.npTrace) < 2:
            print 'Not enough points...'
            return
        self.npTrace = np.array( np.delete(self.npTrace,5,1), dtype = 'float32')[:,0:2]
        self.npTrace = self.npTrace[ self.npTrace[:,0].argsort() ]

        for element in self.npTrace:
            if element[1] > max_val:
                max_val = element[1]
                max_list = np.vstack((max_list,element))

        fig = plt.figure()
        fig.canvas.set_window_title('Ploting Trace')
        if PLOT_LOG: plt.yscale('log')

        plt.plot(self.npTrace[:,0],self.npTrace[:,1],'k.')
        if x_param == 'AMS':
            plt.plot(max_list[:,0],max_list[:,1],'ro')
            plt.rcParams.update({'font.size': 18})
        plt.xlabel('Number of Iterations')
        plt.ylabel('AMS')
        plt.savefig('%s/figure/plot/trace.pdf' % (self.dev_path))
        if show_fig:
            plt.show()
        

    def Plot_Param(self, x_param, show_fig = True):

        if not self.x_param == x_param:
            self.set_data(x_param)

        fig = plt.figure()
        

        mx = max(self.x)
        mn = min(self.x)



        plt.plot(self.x,self.npTrace[:,1],'k.')
        plt.xlabel(self.lbl[x_param])
        plt.ylabel('AMS')
        plt.rcParams.update({'font.size': 18})
        plt.xlim(mn*0.9,mx*1.1)
        plt.yticks(fontsize = 'small')
        fig.canvas.set_window_title('Ploting %s' % x_param)
        fig.suptitle('Ploting %s' % self.lbl[x_param], fontsize=20)
        if self.LOGX:
            x_labels = []
            for el in plt.xticks()[0]:
                x_labels.append( '$10^{%s}$' % str(el) )
            plt.xticks(plt.xticks()[0], x_labels, fontsize = 'small')
            plt.xlim(mn*1.1,mx*0.9)

        plt.savefig('%s/figure/plot/%s.pdf' % (self.dev_path, x_param))
        if show_fig:
            plt.show()
        

    def Plot_Time(self, x_param):

        if not self.x_param == x_param:
            self.set_data(x_param)

        mx = max(self.x)
        mn = min(self.x)


        fig = plt.figure()
        fig.canvas.set_window_title('Ploting %s' % x_param)
        try:
            plt.plot(self.x,self.npTrace[:,14],'k.')  
        except IndexError:
            print 'No entry for time'
            return
        plt.rcParams.update({'font.size': 18})
        if self.LOGX:
            x_labels = []
            for el in plt.xticks()[0]:
                x_labels.append( '$10^{%s}$' % str(el) )
            plt.xticks(plt.xticks()[0], x_labels, fontsize = 'small')
            plt.xlim(mn*1.1,mx*0.9)

        plt.xlabel(x_param)
        plt.ylabel('time')
        plt.show()


    def Plot_Heat(self, x_param, y_param):

        ttl_lbl = {'mask':'Mask', 'seed':5,'nhid':'Hidden Layers','nodes':'Number of Nodes','lrinit':'Learnrate',\
        'lrdecay':'Decay Factor','momentum_init':'Momentum',\
        'momentum_saturate':'Momentum Saturation','momentum_final':'Final Momentum','batch_size':'Batch Size',\
        'loss': 'Loss'}

        self.set_data(x_param , y_param)


        xi, yi = np.linspace(self.x.min(), self.x.max(), 1000), np.linspace(self.y.min(), self.y.max(), 1000)
        xi, yi = np.meshgrid(xi, yi)

        fig = plt.figure()
        zi = scipy.interpolate.griddata((self.x, self.y), self.z, (xi, yi), method='linear')
        plt.xlabel(self.lbl[x_param], fontsize = 'small')
        plt.ylabel(self.lbl[y_param], fontsize = 'small')

        plt.rcParams.update({'font.size': 18})
        imgplot = plt.imshow(zi, vmin=self.z.min(), vmax=self.z.max(), origin='lower',
                   extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()], aspect='auto')
        
        plt.xticks(fontsize = 'small')
        plt.yticks(fontsize = 'small')
        if self.LOGX:
            x_labels = []
            for el in plt.xticks()[0]:
                x_labels.append( '$10^{%s}$' % str(el) )
            plt.xticks(plt.xticks()[0], x_labels)

        if self.LOGY:
            y_labels = []
            for el in plt.yticks()[0]:
                y_labels.append( '$10^{%s}$' % str(el) )
            plt.yticks(plt.yticks()[0], y_labels)

        imgplot.set_cmap('spectral')
        cbar = plt.colorbar()
        cbar.set_label('AMS value')
        plt.axis((self.x.min(),self.x.max(),self.y.min(),self.y.max()))
        plt.savefig('%s/figure/plot/over_%s_%s_clean.pdf' % (self.dev_path, x_param,y_param))
        plt.plot(self.x,self.y,'kx')

        plt.savefig('%s/figure/plot/over_%s_%s.pdf' % (self.dev_path, x_param,y_param))
        plt.show()

    def Plot_All(self,s_name):
        self.Plot_Trace( show_fig = False)
        param = ['seed','nhid','nodes','lrinit','lrdecay','momentum_init','momentum_saturate']
        for arg in param:
            self.Plot_Param(x_param = arg, show_fig = False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-over', help='Plot over', type=str, nargs=2, metavar = ['P1','P2'],default=argparse.SUPPRESS)
    parser.add_argument('-param', help='Plot corresponding AMS value to PARAM', type=str, default=argparse.SUPPRESS)
    parser.add_argument('-trace', help='Plot optimization process', default=argparse.SUPPRESS)
    parser.add_argument('-time', help='Plot optimization process', default=argparse.SUPPRESS)
    parser.add_argument('-sname', help='default', type=str,default='summary.dat')


    args = vars(parser.parse_args())
    objPlot = Plot(s_name = args['sname'])

    if args.has_key('over'):
        x_param = args['over'][0]
        y_param = args['over'][1]
        objPlot.Plot_Heat(x_param=x_param, y_param=y_param)

    if args.has_key('time'):
        objPlot.Plot_Time(x_param =  args['time'])

    if args.has_key('trace'):
        objPlot.Plot_Trace(x_param =  args['trace'])

    if args.has_key('param'):
        objPlot.Plot_Param(x_param = args['param'])