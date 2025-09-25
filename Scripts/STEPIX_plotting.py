# In Scripts/STEPIX_plotting.py
import numpy as np

class SETPIX_extended:
    """
    An extended version of the SETPIX algorithm class that stores all
    intermediate calculation steps as instance attributes, making them
    available for detailed plotting.
    """
    def __init__(self, DataAmps, NoiseThreshold=0.02):
        self.DataAmps = DataAmps
        self.NoiseThreshold = NoiseThreshold
        self._calculate_all_values()

    def _calculate_all_values(self):
        """
        Performs all STEPIX calculations and stores each result (e.g., sorted steps,
        fit parameters, P/Q/R points) as an instance attribute.
        """
        try:
            # --- Initial sorting and step calculation ---
            SortedAmps = np.flip(np.sort(self.DataAmps))
            StepAmps = SortedAmps[:-1] - SortedAmps[1:]
            self.SortedStepAmps = np.flip(np.sort(StepAmps))
            self.StepNumbers = np.arange(1, len(self.SortedStepAmps) + 1)

            # --- D50 Calculation ---
            self.Reconstruction = np.cumsum(self.SortedStepAmps)
            self.Reconstruction_Percentage = 100 * self.Reconstruction / max(self.Reconstruction)
            Amp_50 = max(self.Reconstruction) / 2
            self.D50 = np.nan
            for i in range(len(self.Reconstruction)):
                if self.Reconstruction[i] > Amp_50:
                    self.D50 = i + 1
                    break

            # --- Logarithmic Fit and P Point ---
            # Only use steps above the noise floor for the fit
            StepActivation = self.SortedStepAmps[self.SortedStepAmps > 2.5 * self.NoiseThreshold]
            StepActivationNumbers = np.arange(1, len(StepActivation) + 1)
            
            fit = np.polyfit(np.log(StepActivationNumbers), StepActivation, 1)
            self.log_fit_a = fit[0]
            self.log_fit_b = fit[1]
            self.log_fit_line = self.log_fit_a * np.log(self.StepNumbers) + self.log_fit_b

            self.P_Step = int(np.round(np.exp(-self.log_fit_b / self.log_fit_a)))
            if 0 < self.P_Step <= len(self.SortedStepAmps):
                self.P_Amp = self.SortedStepAmps[self.P_Step - 1]
            else:
                self.P_Step, self.P_Amp = np.nan, np.nan

            # --- Q and R Point Calculation ---
            m = (self.Reconstruction_Percentage[self.P_Step - 1] / self.P_Step 
                 if len(self.Reconstruction_Percentage) > self.P_Step > 0 
                 else 100 / (self.P_Step if self.P_Step > 0 else 1))
            
            self.Q_Step_initial = int(np.round(80 / m))
            self.Qprime_Point = (self.Q_Step_initial, 80)

            Q_Step_current = self.Q_Step_initial
            if len(self.SortedStepAmps) <= Q_Step_current:
                Q_Step_current = len(self.SortedStepAmps)

            # Find the final Q/R point by walking back from the initial Q estimate
            self.Final_Point_Type = "Q"
            while Q_Step_current > 0:
                Q_Amp_current = self.SortedStepAmps[Q_Step_current - 1]
                if Q_Amp_current >= self.NoiseThreshold:
                    break # Found a valid point
                self.Final_Point_Type = "R"
                Q_Step_current -= 1
            
            self.STEPIX = Q_Step_current
            self.Q_Step = self.Q_Step_initial # The 'x' marker is always at the initial Q estimate
            self.Q_Amp = self.SortedStepAmps[self.Q_Step - 1] if self.Q_Step > 0 else np.nan

            self.R_Step = self.STEPIX # The 'o' marker is at the final STEPIX value
            self.R_Amp = self.SortedStepAmps[self.R_Step - 1] if self.R_Step > 0 else np.nan

            self.AMPIX = 1000 * max(self.DataAmps) / self.STEPIX if self.STEPIX > 0 else np.nan

        except Exception as e:
            print(f"Error during STEPIX calculation: {e}")
            attrs = ['SortedStepAmps', 'StepNumbers', 'Reconstruction', 'Reconstruction_Percentage', 'D50', 'log_fit_a', 'log_fit_b', 'log_fit_line', 'P_Step', 'P_Amp', 'Q_Step_initial', 'Qprime_Point', 'STEPIX', 'Q_Step', 'Q_Amp', 'R_Step', 'R_Amp', 'AMPIX', 'Final_Point_Type']
            for attr in attrs:
                setattr(self, attr, np.nan)