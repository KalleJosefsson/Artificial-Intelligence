
import random
import numpy as np

from models import *


#
# Add your Filtering / Smoothing approach(es) here
#Implemented by Kalle
class HMMFilter:
    def __init__(self, probs, tm, om_uf, sm):
        self.__tm = tm
        self.__om_uf = om_uf
        self.__sm = sm
        self.__f_uf = probs
        
        
    
    def filter(self, sensorR):
         
        obs_m_uf = self.__om_uf.get_o_reading(sensorR)
        tm_transposed = self.__tm.get_T_transp()
        f_updated_uf = obs_m_uf @ tm_transposed @ self.__f_uf
        self.__f_uf = f_updated_uf/np.sum(f_updated_uf)
        return self.__f_uf


class HMMFilter_nuf:
    def __init__(self, probs, tm, om_nuf, sm):
        self.__tm = tm
        self.__om_nuf = om_nuf
        self.__sm = sm
        self.__f_nuf = probs
        
        
    def filter_nuf(self,sensorR):
        obs_m_nuf = self.__om_nuf.get_o_reading(sensorR)
        tm_transposed = self.__tm.get_T_transp()
        f_updated_nuf = obs_m_nuf @ tm_transposed @ self.__f_nuf
        self.__f_nuf = f_updated_nuf/np.sum(f_updated_nuf)
        return self.__f_nuf
    
class HMMFilter_uf:
    def __init__(self, probs, tm, om_uf, sm):
        self.__tm = tm
        self.__om_uf = om_uf
        self.__sm = sm
        self.__f_uf = probs
        
        
    
    def filter_uf(self, sensorR):
         
        obs_m_uf = self.__om_uf.get_o_reading(sensorR)
        tm_transposed = self.__tm.get_T_transp()
        f_updated_uf = obs_m_uf @ tm_transposed @ self.__f_uf
        self.__f_uf = f_updated_uf/np.sum(f_updated_uf)
        return self.__f_uf

class HMMFilter_fix_lagged_smoothing:
    def __init__(self, probs, tm, om_nuf, sm):
        self.__tm = tm
        self.__om_nuf = om_nuf
        self.__sm = sm
        self.__lag_steps = 5 
        self.__f_nuf = probs  
        self.__history = []  

    def filter_and_smooth_nuf(self, sensorR):
        
        obs_m_nuf = self.__om_nuf.get_o_reading(sensorR)
        tm_transposed = self.__tm.get_T_transp()
        f_updated_nuf = np.matmul(obs_m_nuf @ tm_transposed, self.__f_nuf)
        self.__f_nuf = f_updated_nuf / np.sum(f_updated_nuf)

    
        self.__history.append((self.__f_nuf.copy(), sensorR, obs_m_nuf))  

        if len(self.__history) > self.__lag_steps:
            smoothed_state = self.__smooth(self.__history[0][0], self.__history[1:])
            self.__history.pop(0) 
            return smoothed_state
        else:
        
            return self.__f_nuf  
        
    
    def __smooth(self, old_belief, observations):
   
        backward_msg = np.ones(self.__sm.get_num_of_states())
        for belief, observation, obs_matrix in reversed(observations):
    
            backward_msg = self.__tm.get_T() @ obs_matrix @  backward_msg
            backward_msg /= np.sum(backward_msg)  

    
        smoothed_belief = old_belief * backward_msg
        smoothed_belief /= np.sum(smoothed_belief)  

        return smoothed_belief
