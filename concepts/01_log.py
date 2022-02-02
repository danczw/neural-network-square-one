'''
What is a log? What does it do?

solving for x (natural log with base eulers number e):
e ** x = b
where b is input value
'''

import numpy as np
import math

b = 5.26

print(np.log(b))
print(math.e ** np.log(b)) # check result - should equal b