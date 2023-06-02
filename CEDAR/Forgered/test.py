import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.array([1,2,3,4])
y = np.array([1,2,6,10])
gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
mn=np.min(x)
mx=np.max(x)
x1=np.linspace(mn,mx,500)
y1=gradient*x1+intercept
plt.plot(x,y,'ob')
plt.plot(x1,y1,'-r')
plt.show()