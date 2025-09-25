import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '2'
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as optimize
from sklearn.cluster import KMeans

class Stairfit:
    def __init__(self, StimData, AmpData, MURange=range(5,251), termination_threshold=0.015, Calc_Thres=False, PlotResult=False, seed=0):
        """
        Initialize the Stairfit class.
        """
        self.StimData = StimData
        self.AmpData = AmpData


        lbLam = min(AmpData)
        ubLam = max(AmpData)

        self.M, self.Lambdas, self.Sizes = self._optimize_lambda(MURange, lbLam, ubLam, termination_threshold)

        self.Diff=0

        if len(self.Sizes)>len(np.unique(self.Sizes)):
            self.Diff=len(self.Sizes)-len(np.unique(self.Sizes))
            print(f"Sizes are non-unique! There are {self.Diff} repititions.")

        if Calc_Thres==True:
            self.rng = np.random.default_rng(seed=seed)
            lbThres= min(StimData)
            ubThres= max(StimData)
            self.Thresholds = self._optimize_threshold(lbThres, ubThres)
            if len(self.Thresholds)>len(np.unique(self.Thresholds)):
                Diff=len(self.Thresholds)-len(np.unique(self.Thresholds))
                print(f"Thresholds are non-unique! There are {Diff} repititions.")
        else:
            self.Thresholds = None
        
        if PlotResult==True:
            self._generate_plot()

    def _optimize_lambda(self, MURange, lbLam, ubLam, termination_threshold):
        """
        Optimize the lambda parameters.
        """
        y = self.AmpData.reshape(-1, 1)
        
        def OptLam(lam, Data):
            Data = Data.reshape(-1, 1)
            diff = np.abs(Data - lam)
            min_diff = np.min(diff, axis=1)
            return np.mean(min_diff)
        
        # Initialize variables
        M = None
        Lambdas = None
        Sizes = None

        for M in MURange:
            #print(M)

            kmeans = KMeans(n_clusters=M+1, init='k-means++', n_init=10, random_state=0)
            kmeans.fit(y)
            lambda_init = kmeans.cluster_centers_

            lambda_init = np.clip(np.sort(lambda_init.reshape(-1)),lbLam,ubLam)

            Bounds = ([lbLam, ubLam],)*(M+1)
            result = optimize.minimize(OptLam, lambda_init, args=(self.AmpData), method='Nelder-Mead', bounds=Bounds)
            Lambdas = np.sort(result.x)
            Error = OptLam(Lambdas, self.AmpData)
            if Error < termination_threshold or M>250:
                diff=[]
                for i in range(len(Lambdas)-1):
                    diff.append(Lambdas[i+1]-Lambdas[i])
                Sizes=np.array(diff)
                break
            
         # Check if Sizes, Lambdas, and M were assigned
        if Sizes is None or Lambdas is None or M is None:
            raise ValueError("Failed to optimize lambda parameters within the given range")
           
        return M, Lambdas, Sizes

    def _optimize_threshold(self, lb, ub):
        """
        Optimize the threshold parameters.
        """
        x = self.StimData
        y = self.AmpData
        lam = self.Lambdas

        thr_init = self.rng.uniform(lb,ub,self.M)

        Bounds = ([lb,ub],)*(self.M)

        Datax = x.reshape(-1,1)
        Datay = y.reshape(-1,1)
        newlams = np.repeat(lam,2)[1:-1]

        def OptThr(thr):
            thr = np.sort(thr)
            newthr = np.repeat(thr,2)
            A = self._Dm(Datax, Datay, newthr, newlams)
            index = np.searchsorted(thr, x, side='left')
            centerlams = lam[index]
            B = self._ydist(y, centerlams)
            ManDis = np.c_[A,B]
            min_diff = np.min(ManDis, axis=1)
            return np.sum(min_diff)

        result2 = optimize.minimize(OptThr, thr_init, method='Powell', bounds=Bounds, options={'disp': True, 'xtol': 1e-07})

        Thresholds = np.sort(result2.x)
        Thresholds = np.concatenate([[lb],Thresholds,[ub]])
        
        return Thresholds

    def _generate_plot(self):
        """
        Generate the plot.
        """
        StimData = self.StimData
        AmpData = self.AmpData
        Lambdas = self.Lambdas
        Thresholds = self.Thresholds
        Sizes = self.Sizes

        newlams = np.repeat(Lambdas,2)
        newthr = np.repeat(Thresholds,2)[1:-1]
        

        SizePlot = np.repeat(Sizes,2)
        ThreshPlot = np.repeat(Thresholds[1:-1],2)
        
        Fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        Fig.add_trace(go.Scatter(x=StimData, y=AmpData, marker=dict(color="#FF0000",size=4), mode='markers', name='Data'), row=2,col=1)
        Fig.add_trace(go.Scatter(x=newthr, y=newlams, line=dict(color="Black", width=3), mode='lines', name='Stairfit'), row=2,col=1)

        i=0
        while i<len(SizePlot):
            Fig.add_trace(go.Scatter(x=ThreshPlot[i:i+2], y=[0,SizePlot[i]], line=dict(width=2.5), mode='lines'), row=1,col=1)
            i+=2        

        Fig.update_xaxes(title_text="Threshold (mA)", row=1, col=1)
        Fig.update_xaxes(title_text="Stimulus (mA)", row=2, col=1)
        Fig.update_yaxes(title_text="Amplitude (mV)", row=1, col=1)
        Fig.update_yaxes(title_text="Amplitude (mV)", row=2, col=1)

        Fig.show()

    def _Dm(self,datax, datay, modelx, modely):
        """
        Calculate Manhattan distance between data and model.
        """
        return 0.1*np.abs(datax-modelx) + np.abs(datay-modely)

    def _ydist(self,datay, lam):
        """
        Calculate absolute distance between data y and lambda.
        """
        return np.abs(datay-lam)
