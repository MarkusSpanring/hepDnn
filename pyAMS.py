#TAMS
#Author: Martin Flechl, 2014
#calculate average median significance of two histograms
###############################################################################
#                  Edited by: Markus Spanring HEPHY Vienna 2015               #
###############################################################################
#Transferred C-code to python code
#
import os
import numpy as np
import sys
import ROOT



class TAMS():
    def __init__(self, hSig, hBg, rsys=0.2, br = 1.):

        ROOT.gROOT.Reset()

        m_simple = np.full(3,0) #simple s/sqrt(b)
        m_simple_syst = np.full(3,0) #simple s/sqrt(b(1+sys*sys*b))
        m_simple_stat = np.full(3,0) #simple s/sqrt(b) with stat. unc. on b
        m_simple_syst_stat = np.full(3,0) #simple s/sqrt(b(1+sys*sys*b)) with stat unc. on b
        m_ams = np.full(3,0) #ams #0 central, 1 -err, 2 +err
        m_ams_stat = np.full(3,0) #ams with stat unc on b
        m_ams_syst = np.full(3,0) #ams with sys unc on b
        m_ams_syst_stat = np.full(3,0) #ams with sys unc on b

        

        self._m_h1 = hSig
        self._m_h2 = hBg
        self._m_h1s = None
        self._m_h2s = None
        self._m_rsys = float(rsys)
        self.bmin=float(0.1)
        self.br = float(br)
        self.ER = False 
        self._m_simple = np.array(m_simple, dtype='float32')
        self._m_simple_syst = np.array(m_simple_syst, dtype='float32')
        self._m_simple_stat = np.array(m_simple_stat, dtype='float32')
        self._m_simple_syst_stat = np.array(m_simple_syst_stat, dtype='float32')
        self._m_ams = np.array(m_ams, dtype='float32')
        self._m_ams_stat = np.array(m_ams_stat, dtype='float32')
        self._m_ams_syst = np.array(m_ams_syst, dtype='float32')
        self._m_ams_syst_stat = np.array(m_ams_syst_stat, dtype='float32')

        self._m_simple[0]=-1
        self._m_simple_syst[0]=-1
        self._m_simple_stat[0]=-1
        self._m_simple_syst_stat[0]=-1
        self._m_ams[0]=-1
        self._m_ams_stat[0]=-1
        self._m_ams_syst[0]=-1
        self._m_ams_syst_stat[0]=-1

    def seth1(self,hSig):
        del self._m_h1
        self._m_h1 = hSig
    def seth2(self,hBg):
        del self._m_h2
        self._m_h2 = hBg
    def seth(self,hSig,hBg):
        del self._m_h1
        del self._m_h2
        self._m_h1 = hSig
        self._m_h2 = hBg
    def setsys(self,rsys): self._m_rsys = rsys

    def _any(self,m, o):

        #if (m<0): self.calc()
        self.calc(br = self.br)
        if (o==0): return m[o]
        elif (o==-1): return m[1]
        elif (o==1): return m[2]
        else: return -1

    def ams(self,o=0):
        o=int(o)
        m_any=self._m_ams
        return self._any(m_any, o)

    def ams_stat(self,o=0):
        o=int(o)
        m_any=self._m_ams_stat
        return self._any(m_any, o)

    def ams_syst(self,o=0):
        o=int(o)
        m_any=self._m_ams_syst
        return self._any(m_any, o)

    def ams_syst_stat(self,o=0):
        o=int(o)
        m_any=self._m_ams_syst_stat
        return self._any(m_any, o)

    def simple(self,o=0):
        o=int(o)
        m_any=self._m_simple
        return self._any(m_any, o)

    def simple_stat(self,o=0):
        o=int(o)
        m_any=self._m_simple_stat
        return self._any(m_any, o)

    def simple_syst(self,o=0):
        o=int(o)
        m_any=self._m_simple_syst
        return self._any(m_any, o)

    def simple_syst_stat(self,o=0):
        o=int(o)
        m_any=self._m_simple_syst_stat
        return self._any(m_any, o)

    def listAll(self):
        strMsg  =",".join(["Systematics: " , str(self._m_rsys)+"\n"])
        strMsg +=",".join(["AMS              :" , str(self.ams(0)), "\t -" ,str(self.ams(-1)), " +" ,str(self.ams(1))+"\n"])
        strMsg +=",".join(["AMS    stat      :" , str(self.ams_stat(0)), "\t -" ,str(self.ams_stat(-1)), " +" , str(self.ams_stat(1))+"\n"])
        strMsg +=",".join(["AMS    syst      :" , str(self.ams_syst(0)), "\t -" , str(self.ams_syst(-1)), " +" , str(self.ams_syst(1))+"\n"])
        strMsg +=",".join(["AMS    stat syst :" , str(self.ams_syst_stat(0)), "\t -" , str(self.ams_syst_stat(-1)), " +" , str(self.ams_syst_stat(1))+"\n"])
        strMsg +=",".join(["Simple              :" , str(self.simple(0)), "\t -" ,str(self.simple(-1)), " +" ,str(self.simple(1))+"\n"])
        strMsg +=",".join(["Simple    stat      :" , str(self.simple_stat(0)), "\t -" ,str(self.simple_stat(-1)), " +" , str(self.simple_stat(1))+"\n"])
        strMsg +=",".join(["Simple    syst      :" , str(self.simple_syst(0)), "\t -" , str(self.simple_syst(-1)), " +" , str(self.simple_syst(1))+"\n"])
        strMsg +=",".join(["Simple    stat syst :" , str(self.simple_syst_stat(0)), "\t -" , str(self.simple_syst_stat(-1)), " +" , str(self.simple_syst_stat(1))+"\n"])

        return strMsg

    def savePlot(self,fname="plot_tams.png"):

        dev_path = os.environ['DNN_PATH']

        if not os.path.exists("%s/figure" % dev_path):
            os.makedirs("%s/figure" % dev_path)

        if not os.path.exists("%s/figure/hist" % dev_path):
            os.makedirs("%s/figure/hist" % dev_path)

        if not self.ER:
          self._m_h1s=ROOT.TH1F(self._m_h1)
          self._m_h2s=ROOT.TH1F(self._m_h2)

        ROOT.gStyle.SetOptStat(0)

        self._m_h2s.SetXTitle("DNN score")
        self._m_h2s.SetYTitle("Events")
        self._m_h1s.SetLineColor(ROOT.kRed)
        self._m_h2s.SetLineColor(ROOT.kBlue)
        
        #ROOT.TH1F("s","",nbins,blow, bhigh )
        m_h3=ROOT.TH1F(self._m_h2)
        m_h2b=ROOT.TH1F(self._m_h2s)
        for i in xrange(m_h3.GetNbinsX()+2):
            m_h3.SetBinError(i,0);

        m_h3.Add(self._m_h1s);
        m_h3.SetMarkerSize(0.9);
        m_h3.SetLineWidth(3);
        self._m_h2s.SetFillColor(ROOT.kBlack);
        self._m_h2s.SetFillStyle(3004);
        self._m_h2s.SetMarkerSize(0.0);

        m_h2b.SetLineColor(ROOT.kOrange+1);
        m_h2b.SetFillColor(ROOT.kOrange+1);

        leg=ROOT.TLegend(0.65,0.75,0.85,0.85);
        leg.AddEntry(self._m_h1s,"Signal","lep");
        leg.AddEntry(self._m_h2s,"Background","lep");
        leg.SetFillColor(10);
        leg.SetShadowColor(10);
        leg.SetLineColor(10);

        leg2=ROOT.TLegend(0.45,0.7,0.85,0.85);
        leg2.AddEntry(m_h2b,"Background","fe");
        leg2.AddEntry(m_h3,"Signal+Background","lep");
        leg2.SetFillColor(10);
        leg2.SetShadowColor(10);
        leg2.SetLineColor(10);

        cx=ROOT.TCanvas("cx","cx");
        self._m_h2s.Draw();
        self._m_h1s.Draw("same");
        ROOT.gPad.SetLogy();
        leg.Draw();
        ROOT.gPad.SaveAs("%s/figure/hist/%s" % (dev_path,fname));

        cy=ROOT.TCanvas("cy","cy");
        m_h2b.Draw("hist");
        self._m_h2s.Draw("E2same");
        m_h3.Draw("same");
        ROOT.gPad.SetLogy();
        leg2.Draw();

        t=ROOT.TLatex( 0.5, 0.88, " " );


        t.SetNDC();
        t.SetTextSize(0.035);

        t.Draw()
        fname = fname.replace('rebin', 'hist')
        ROOT.gPad.SaveAs("%s/figure/hist/%s" % (dev_path,fname))

        cx.Close()
        cy.Close()

        del m_h3
        del m_h2b
        del self._m_h1s
        del self._m_h2s


    def rebinEqui(self):

        self.rebin()
        
        nbins=int(self._m_h1.GetNbinsX())

        bin_contents_s=np.full(nbins+2,0)
        bin_contents_b=np.full(nbins+2,0)
        bin_errors_s=np.full(nbins+2,0)
        bin_errors_b=np.full(nbins+2,0)

        for j in xrange(nbins+2):
            bin_contents_s[j]=self._m_h1.GetBinContent(j)
            bin_contents_b[j]=self._m_h2.GetBinContent(j)
            bin_errors_s[j]=self._m_h1.GetBinError(j)
            bin_errors_b[j]=self._m_h2.GetBinError(j)


        blow=float(self._m_h1.GetBinLowEdge(1))
        bhigh=float(self._m_h1.GetBinLowEdge(nbins+1))

        
        del self._m_h1
        del self._m_h2
        del self._m_h1s
        del self._m_h2s


        self._m_h1=ROOT.TH1F("s","",nbins,blow, bhigh )
        self._m_h2=ROOT.TH1F("b","",nbins,blow, bhigh )
        self._m_h1s=ROOT.TH1F("ss","",nbins,blow, bhigh )
        self._m_h2s=ROOT.TH1F("bs","",nbins,blow, bhigh )




        self._m_h1.SetContent(bin_contents_s)
        self._m_h2.SetContent(bin_contents_b)
        self._m_h1s.SetContent(bin_contents_s)
        self._m_h2s.SetContent(bin_contents_b)
        self._m_h1.SetError(bin_errors_s)
        self._m_h2.SetError(bin_errors_b)


        
        for i in xrange(nbins+2):
            bin_errors_s[i]=np.sqrt( (bin_errors_s[i])**2 + (bin_contents_s[i]*self._m_rsys)**2 )
            bin_errors_b[i]=np.sqrt( (bin_errors_b[i])**2 + (bin_contents_b[i]*self._m_rsys)**2 )

        self._m_h1s.SetError(bin_errors_s)
        self._m_h2s.SetError(bin_errors_b)


        del bin_contents_s
        del bin_contents_b
        del bin_errors_s
        del bin_errors_b


        self.ER=True

  

    def rebin(self):
        #    const float RELSTATMAX=0.5
        RELSTATMAX=float(0.5)
        BINC=float(1.4)

        bin_edge = np.array([])
        bin_s = np.array([])
        bin_b = np.array([])
        bin_berr = np.array([])
        bin_serr = np.array([])
        
        nedges=int(self._m_h1.GetNbinsX()+1) #edges=bins+1
        highest_edge=float(self._m_h1.GetBinLowEdge( nedges ))
        highest_edge = np.array([highest_edge])
        bin_edge = np.append(bin_edge, highest_edge)

        

        s=float(0)
        b=float(0)
        berr2=float(0)
        serr2=float(0)
        bprev=float(0)
        for i in reversed(xrange(1,nedges-1)): #loop over bin edges

            s+=self._m_h1.GetBinContent(i)
            b+=self._m_h2.GetBinContent(i)
            serr2+=self._m_h1.GetBinError(i)**2
            berr2+=self._m_h2.GetBinError(i)**2

            t_edge=self._m_h1.GetBinLowEdge( i )
            #check if this is a new edge
            if ( b<1e-3 ): continue #if b is negativ or 0 or very small, continue
            if ( (np.sqrt(berr2)/b)>RELSTATMAX ): continue #if the rel stat unc on the background is >X%, continue
            if ( b<bprev*BINC ): continue #more b than bin to the right (previous bin)


            if ( t_edge<0.8 ):
                if ( bin_edge[-1]-t_edge < 0.05 ): continue
            if ( t_edge<0.6 ):
                if ( bin_edge[-1]-t_edge < 0.10 ): continue
            if ( t_edge<0.4 ):
                if ( bin_edge[-1]-t_edge < 0.20 ): continue

            #we have a new edge!
            bin_edge = np.append(bin_edge,self._m_h1.GetBinLowEdge( i ) )
            bin_s = np.append(bin_s,s )
            bin_b = np.append(bin_b,b )
            bin_serr = np.append(bin_serr,np.sqrt(serr2) )
            bin_berr = np.append(bin_berr,np.sqrt(berr2) )
            
            bprev=b
            b=0
            s=0
            serr2=0
            berr2=0

        
        #    lbin=bin_edge.size()-1
        if (  np.abs( bin_edge[-1]-self._m_h1.GetBinLowEdge( 1 )  )>1e-5  ): #replace lowest bin boarder with lower-most old bin border
            bin_edge[-1]=self._m_h1.GetBinLowEdge( 1 )
            bin_s[-1]+=s
            bin_b[-1]+=b
            bin_serr[-1]=np.sqrt( np.power( bin_serr[-1], 2 )+serr2 )
            bin_berr[-1]=np.sqrt( np.power( bin_berr[-1], 2 )+berr2 )
        

        #Create new histos
        nbins=int(bin_edge.shape[0]-1)
        array_edge=np.full(nbins+1,0)

        for j in xrange(nbins+1):
            array_edge[j]=bin_edge[nbins-j]

        self._m_h1=ROOT.TH1F("s","",nbins,array_edge)
        self._m_h2=ROOT.TH1F("b","",nbins,array_edge)



        for j in xrange(nbins):
            self._m_h1.SetBinContent(j+1,bin_s[nbins-1-j] )
            self._m_h1.SetBinError(  j+1,bin_serr[nbins-1-j] )
            self._m_h2.SetBinContent(j+1,bin_b[nbins-1-j] )
            self._m_h2.SetBinError(  j+1,bin_berr[nbins-1-j] )

        

    def calc(self,br=1.0): 
        nbins=int(self._m_h1.GetNbinsX())
        if ( nbins != self._m_h2.GetNbinsX() ):
          print "ERROR: Not the same number of bins! Giving up...\n" 
          return
        
        if ( self._m_h1.GetBinLowEdge(nbins) != self._m_h2.GetBinLowEdge(nbins) ):
          print "ERROR: Not the same range! Giving up...\n" 
          return
        self.br = br

        #float s, b, bt, b_stat, b_syst, b_stat2, b_syst2, s_stat
        self._m_ams[0]=0
        self._m_ams_stat[0]=0
        self._m_ams_syst[0]=0
        self._m_ams_syst_stat[0]=0
        self._m_simple[0]=0
        self._m_simple_syst[0]=0
        self._m_simple_stat[0]=0
        self._m_simple_syst_stat[0]=0
        self.br = float(br)

        _ams_err = np.full(4,0)
        _ams_stat_err = np.full(4,0)
        _ams_syst_err = np.full(4,0)
        _ams_syst_stat_err = np.full(4,0)
        _simple_err = np.full(4,0)
        _simple_syst_err = np.full(4,0)
        _simple_stat_err = np.full(4,0)
        _simple_syst_stat_err = np.full(4,0) 

        for ibin in xrange(1,self._m_h1.GetNbinsX()+1):
            s=self._m_h1.GetBinContent(ibin)
            b=self._m_h2.GetBinContent(ibin)
            s_stat=self._m_h1.GetBinError(ibin) #absolute unc
            b_stat=self._m_h2.GetBinError(ibin) #absolute unc
            if ( b_stat<0.5*np.sqrt(br) ): b_stat=0.5*np.sqrt(br)
            b_stat2=b_stat*b_stat
            b_syst=b*self._m_rsys #absolute unc
            if (b_syst<self._m_rsys*br ): b_syst=self._m_rsys*br
            b_syst2=b_syst*b_syst
            bt=b+br

            self._m_simple[0]           += self.calc_simple2(s, bt)
            self._m_simple_stat[0]      += self.calc_simple2(s, bt, b_stat2)
            self._m_simple_syst[0]      += self.calc_simple2(s, bt, b_syst2)
            self._m_simple_syst_stat[0] += self.calc_simple2(s, bt, b_stat2+b_syst2)

            self._m_ams[0]              += self.calc_ams2(s, bt)
            self._m_ams_stat[0]         += self.calc_ams2(s, bt, b_stat2)
            self._m_ams_syst[0]         += self.calc_ams2(s, bt, b_syst2)
            self._m_ams_syst_stat[0]    += self.calc_ams2(s, bt, b_syst2+b_stat2)

            self.calc_simple2_err(_simple_err          , s, bt, s_stat, b_stat)
            self.calc_simple2_err(_simple_stat_err     , s, bt, s_stat, b_stat, b_stat2)
            self.calc_simple2_err(_simple_syst_err     , s, bt, s_stat, b_stat, b_syst2)
            self.calc_simple2_err(_simple_syst_stat_err, s, bt, s_stat, b_stat, b_syst2+b_stat2)

            self.calc_ams2_err(_ams_err          , s, bt, s_stat, b_stat)
            self.calc_ams2_err(_ams_stat_err     , s, bt, s_stat, b_stat, b_stat2)
            self.calc_ams2_err(_ams_syst_err     , s, bt, s_stat, b_stat, b_syst2)
            self.calc_ams2_err(_ams_syst_stat_err, s, bt, s_stat, b_stat, b_syst2+b_stat2)

          #      if ( ( m_h1->GetNbinsX()-ibin )<=3 ){
          #        cout << "Bin " << ibin << ":\t AMS=" << sqrt(calc_ams2(s, bt, b_syst2+b_stat2)) << "\t total=" << sqrt(m_ams_syst_stat[0]) << "\t s=" << s << " , b=" << b << endl
          #        cout << "Bin " << ibin << ":\t AMS=" << sqrt(calc_simple2(s, bt, b_syst2+b_stat2)) << "\t total=" << sqrt(m_simple_syst_stat[0]) << endl
          #      }

        
        self._m_simple[0]=np.sqrt(self._m_simple[0])
        self._m_simple_stat[0]=np.sqrt(self._m_simple_stat[0])
        self._m_simple_syst[0]=np.sqrt(self._m_simple_syst[0])
        self._m_simple_syst_stat[0]=np.sqrt(self._m_simple_syst_stat[0])

        self._m_ams[0]=np.sqrt(self._m_ams[0])
        self._m_ams_stat[0]=np.sqrt(self._m_ams_stat[0])
        self._m_ams_syst[0]=np.sqrt(self._m_ams_syst[0])
        self._m_ams_syst_stat[0]=np.sqrt(self._m_ams_syst_stat[0])

        self._m_ams[1]= np.sqrt(   ( np.sqrt(_ams_err[0])-self._m_ams[0])**2  + ( np.sqrt(_ams_err[3])-self._m_ams[0])**2 )
        self._m_ams[2]= np.sqrt(   ( np.sqrt(_ams_err[1])-self._m_ams[0])**2  + ( np.sqrt(_ams_err[2])-self._m_ams[0])**2 )
        self._m_ams_stat[1]= np.sqrt(   ( np.sqrt(_ams_stat_err[0])-self._m_ams_stat[0])**2  + ( np.sqrt(_ams_stat_err[3])-self._m_ams_stat[0])**2 )
        self._m_ams_stat[2]= np.sqrt(   ( np.sqrt(_ams_stat_err[1])-self._m_ams_stat[0])**2  + ( np.sqrt(_ams_stat_err[2])-self._m_ams_stat[0])**2 )
        self._m_ams_syst[1]= np.sqrt(   ( np.sqrt(_ams_syst_err[0])-self._m_ams_syst[0])**2  + ( np.sqrt(_ams_syst_err[3])-self._m_ams_syst[0])**2 )
        self._m_ams_syst[2]= np.sqrt(   ( np.sqrt(_ams_syst_err[1])-self._m_ams_syst[0])**2  + ( np.sqrt(_ams_syst_err[2])-self._m_ams_syst[0])**2 )
        self._m_ams_syst_stat[1]= np.sqrt(   ( np.sqrt(_ams_syst_stat_err[0])-self._m_ams_syst_stat[0])**2  + ( np.sqrt(_ams_syst_stat_err[3])-self._m_ams_syst_stat[0])**2 )
        self._m_ams_syst_stat[2]= np.sqrt(   ( np.sqrt(_ams_syst_stat_err[1])-self._m_ams_syst_stat[0])**2  + ( np.sqrt(_ams_syst_stat_err[2])-self._m_ams_syst_stat[0])**2 )

        self._m_simple[1]= np.sqrt(   ( np.sqrt(_simple_err[0])-self._m_simple[0])**2  + ( np.sqrt(_simple_err[3])-self._m_simple[0])**2 )
        self._m_simple[2]= np.sqrt(   ( np.sqrt(_simple_err[1])-self._m_simple[0])**2  + ( np.sqrt(_simple_err[2])-self._m_simple[0])**2 )
        self._m_simple_stat[1]= np.sqrt(   ( np.sqrt(_simple_stat_err[0])-self._m_simple_stat[0])**2  + ( np.sqrt(_simple_stat_err[3])-self._m_simple_stat[0])**2 )
        self._m_simple_stat[2]= np.sqrt(   ( np.sqrt(_simple_stat_err[1])-self._m_simple_stat[0])**2  + ( np.sqrt(_simple_stat_err[2])-self._m_simple_stat[0])**2 )
        self._m_simple_syst[1]= np.sqrt(   ( np.sqrt(_simple_syst_err[0])-self._m_simple_syst[0])**2  + ( np.sqrt(_simple_syst_err[3])-self._m_simple_syst[0])**2 )
        self._m_simple_syst[2]= np.sqrt(   ( np.sqrt(_simple_syst_err[1])-self._m_simple_syst[0])**2  + ( np.sqrt(_simple_syst_err[2])-self._m_simple_syst[0])**2 )
        self._m_simple_syst_stat[1]= np.sqrt(   ( np.sqrt(_simple_syst_stat_err[0])-self._m_simple_syst_stat[0])**2  + ( np.sqrt(_simple_syst_stat_err[3])-self._m_simple_syst_stat[0])**2 )
        self._m_simple_syst_stat[2]= np.sqrt(   ( np.sqrt(_simple_syst_stat_err[1])-self._m_simple_syst_stat[0])**2  + ( np.sqrt(_simple_syst_stat_err[2])-self._m_simple_syst_stat[0])**2 )


    def calc_simple2(self,s,b,berr2=0.):
        #if ( (b+berr2)<1e-8 ):
            #print "XX ", b , " + ", berr2
        if (b<self.bmin): b=self.bmin
        return np.power( ( s/np.sqrt( b + berr2 ) ) , 2 )
      

      
    def calc_ams2(self,s,b,berr2=-1):
        if (b<self.bmin): b=self.bmin;
        if (berr2<0.): return 2*(   (s+b) * np.log( 1 + s/b  ) - s  )
        return 2*((s+b)*np.log((s+b)*(b+berr2)/(b*b+(s+b)*berr2))-b*b/berr2*np.log(1+berr2*s/(b*(b+berr2))))
      

    def calc_ams2_err(self,m_ams_err,s,b,s_stat,b_stat,berr2=-1.):
        m_ams_err[0]           += self.calc_ams2(s, b+b_stat, berr2)
        m_ams_err[1]           += self.calc_ams2(s, b-b_stat, berr2)
        m_ams_err[2]           += self.calc_ams2(s+s_stat, b, berr2)
        m_ams_err[3]           += self.calc_ams2(s-s_stat, b, berr2)
      
    def calc_simple2_err(self,m_simple_err,s,b,s_stat,b_stat,berr2=0.):
        m_simple_err[0]           += self.calc_simple2(s, b+b_stat, berr2)
        m_simple_err[1]           += self.calc_simple2(s, b-b_stat, berr2)
        m_simple_err[2]           += self.calc_simple2(s+s_stat, b, berr2)
        m_simple_err[3]           += self.calc_simple2(s-s_stat, b, berr2)