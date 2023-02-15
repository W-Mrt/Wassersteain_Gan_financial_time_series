import pandas as pd
import numpy as np
import arch


#-------------GARCH fitting
def fill_garch(self,ret,validation_split=0.2):
		c = 100
		am = arch.arch_model(ret[:int((1-validation_split)*ret.size)]*c)
		res = am.fit(update_freq=5,disp='on')
		new_cond_vol = np.zeros(ret.size)
		c_vol = res.conditional_volatility
		for i in range(ret.size):
			if c_vol.size > i:
				new_cond_vol[i] = c_vol[i]
			else:
				new_cond_vol[i] = np.sqrt(res.params['omega'] + res.params['alpha[1]'] * ((ret[i-1]*c)-res.params['mu'])**2 + (new_cond_vol[i - 1]**2)*res.params['beta[1]'])
		return new_cond_vol/c