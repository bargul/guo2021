# BUNDAN BISEY CIKMADI BIRAKTIM , LT ICINDI
import numpy as np
import matplotlib.pyplot as plt
  
lower = 4  # the lower bound for your values
shape = 6   # the distribution shape parameter, also known as `a` or `alpha`
size = 1142 # the size of your sample (number of random values)

s = (np.random.pareto(shape, size) + 1) * lower

count, bins, _ = plt.hist(s,100,density=True)
plt.show()