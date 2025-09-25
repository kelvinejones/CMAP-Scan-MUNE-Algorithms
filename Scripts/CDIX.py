import numpy as np
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture as GMM
import plotly.graph_objects as go
import scipy.signal as signal
from scipy.stats import norm

class CDIX:
    def __init__(self, DataStims, DataAmps, Mean_LS=1.75, PlotResult=False):
        """
        Initialize the CDIX class.
        """
        self.DataStims = DataStims
        self.DataAmps = DataAmps

        self.G, self.NDivs, self.CDIX, self.S, self.D = self._calculate_CDIX(Mean_LS)

        if PlotResult==True:   
            self._generate_plot()


    def _calculate_CDIX(self,Mean_LS):
        try:
            self.MidStims, self.MidAmps = self._segment_data()
            S = self._GMM(Mean_LS)
            G, NDivs, CDIX, D = self._find_CDIX_params(S)
        except:
            return np.nan, np.nan, np.nan, S, np.nan
        return G, NDivs, CDIX, S, D



    def _segment_data(self):

        """Segmentation Step"""


        Index=np.argwhere(self.DataStims==np.amax(self.DataStims))[-1][0]
        DataStims=np.flip(self.DataStims[Index:])
        DataAmps=np.flip(self.DataAmps[Index:])
        N=len(DataStims)
        NFc=50/N
        b,a=signal.butter(3, NFc, btype='low', analog=False)
        E=signal.filtfilt(b,a,DataAmps)
        self.E=E

        t1=0.02
        t2=0.95

        lowbound=min(E)+(max(E)-min(E))*t1
        highbound=min(E)+(max(E)-min(E))*t2

        for i in range(len(E)):
            if E[i] < lowbound:
                self.prescan_index=i
            if E[i] > highbound:
                self.postscan_index=i+1
                break
        
        MidStims=DataStims[self.prescan_index:self.postscan_index]
        MidAmps=DataAmps[self.prescan_index:self.postscan_index]

        return MidStims,MidAmps
    
    def _GMM(self, Mean_LS):
        """Gaussian Mixed Modeling Step"""
        MidAmps=self.MidAmps

        S=np.array([])
        for i in range(len(MidAmps)-1):
            S=np.append(S,abs(MidAmps[i]-MidAmps[i+1]))


        S_SS=np.append(S,-S)

        Std_LS=(1/3)*Mean_LS
        Std_SS=(1/5)*Std_LS

        means_init = np.array([Mean_LS, -Mean_LS, 0]).reshape(-1, 1) # replace with your initial means

        # initial standard deviations
        std_init = np.array([Std_LS, Std_LS, Std_SS]) # replace with your initial standard deviations

        # initial weights
        weights_init = np.array([0.33, 0.33, 0.34]) # REMOVE AND TEST LATER

        # initial precisions (inverse of variance)
        precisions_init = 1 / np.square(std_init).reshape(-1, 1, 1)

        # model creation
        gmm = GMM(n_components=3, 
                            means_init=means_init, 
                            precisions_init=precisions_init, 
                            weights_init=weights_init, 
                            tol=0.01) 

        # fit the model
        gmm.fit(S_SS.reshape(-1,1))

        # get the final means
        final_means = gmm.means_

        # get the final standard deviations
        final_std_dev = np.sqrt(gmm.covariances_)

        # get the final weights
        self.GMM_weights = gmm.weights_

        self.GMM_means = final_means.ravel()
        self.GMM_std_dev = final_std_dev.ravel()

        return S
    
    def _find_CDIX_params(self,S):
        # G_LS=self.GMM_weights[0]*norm.pdf(S, self.GMM_means[0], self.GMM_std_dev[0])
        # G_SS=self.GMM_weights[2]*norm.pdf(S, self.GMM_means[2], self.GMM_std_dev[2])
        # bools=G_LS>G_SS

        # try:
        #     G=min(S[bools])
        # except:
        # X_int=np.linspace(0,10*max(S),10000)
        # G_LS=self.GMM_weights[0]*norm.pdf(X_int, self.GMM_means[0], self.GMM_std_dev[0])
        # G_SS=self.GMM_weights[2]*norm.pdf(X_int, self.GMM_means[2], self.GMM_std_dev[2])
        # bools=G_LS>G_SS
        l=self._solve_intersect(self.GMM_means[0],self.GMM_means[2],self.GMM_std_dev[0],self.GMM_std_dev[2],self.GMM_weights[0],self.GMM_weights[2])
        while True:
            if l.size<=0:
                print('bruh')
                self.GMM_weights[0]*=2
                l=self._solve_intersect(self.GMM_means[0],self.GMM_means[2],self.GMM_std_dev[0],self.GMM_std_dev[2],self.GMM_weights[0],self.GMM_weights[2])
            else:
                break

        G=min([i for i in l if i > 0])
        #G=min(X_int[bools])
        D=np.array([])

        for i in self.MidAmps:
            D=np.append(D,np.floor(i/G))
        L=len(D)
        NDiv=max(D)-min(D)+1

        H_arr=np.array([])
        for i in np.arange(min(D),max(D)+1):
            Fi=np.count_nonzero(D==i)
            if Fi>0:
                H_arr=np.append(H_arr,(Fi/L)*np.log2(Fi/L))


        H=-np.sum(H_arr)
        CDIX=2**H

        return G,NDiv,CDIX,D

    def _solve_intersect(self,m1,m2,std1,std2,s1,s2):
        a = 1/(2*std1**2) - 1/(2*std2**2)
        b = m2/(std2**2) - m1/(std1**2)
        c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log((std2*s1)/(std1*s2))
        r=np.roots([a,b,c])
        return r[np.isreal(r)]

    def _generate_plot(self):

        Index=np.argwhere(self.DataStims==np.amax(self.DataStims))[-1][0]
        DataStims=np.flip(self.DataStims[Index:])
        DataAmps=np.flip(self.DataAmps[Index:])
        MidStims=self.MidStims
        MidAmps=self.MidAmps
        E=self.E
        G=self.G
        S=self.S
        D=self.D

        x=np.linspace(0,2*max(S),1000)

        y1=len(S)*0.08*self.GMM_weights[0]*norm.pdf(x, self.GMM_means[0], self.GMM_std_dev[0])
        y2=len(S)*0.08*self.GMM_weights[1]*norm.pdf(x, self.GMM_means[1], self.GMM_std_dev[1])
        y3=len(S)*0.08*self.GMM_weights[2]*norm.pdf(x, self.GMM_means[2], self.GMM_std_dev[2])

        Fig=make_subplots(
                rows=1, cols=2, subplot_titles=("Data", "GMM")
            )

        Fig.add_trace(go.Scatter(x=DataStims, y=DataAmps, marker=dict(color="#FF0000",size=3), mode='markers', name="RawData"), row=1, col=1)
        Fig.add_trace(go.Scatter(x=DataStims, y=E,mode='lines',name="SmoothedData"), row=1, col=1)
        Fig.add_trace(go.Scatter(x=MidStims, y=MidAmps, marker=dict(color="#FF00FF",size=3),mode='markers', name="MidLineData"), row=1, col=1)

        for y in np.arange(min(D),max(D)+2):
            Fig.add_hline(y=y*G,line_width=0.1, line_dash="dash", line_color="black", row=1, col=1)

        xbins=dict(size=0.04)
        Fig.add_trace(go.Histogram(x=S,xbins=xbins, name="Step Size"), row=1, col=2)
        Fig.add_trace(go.Scatter(x=x,y=y1,mode='lines',marker=dict(color="#FF0000",size=3), name="LS Comp"), row=1, col=2)
        Fig.add_trace(go.Scatter(x=x,y=y2,mode='lines',marker=dict(color="#0000FF",size=3), name="MirrorLS Comp"), row=1, col=2)
        Fig.add_trace(go.Scatter(x=x,y=y3,mode='lines',marker=dict(color="#00FF00",size=3), name="SS Comp"), row=1, col=2)

        Fig.add_vline(x=G, line_width=3, line_dash="dash", line_color="green", name=f"CDIX ({CDIX})",row=1, col=2)
        Fig.show()

