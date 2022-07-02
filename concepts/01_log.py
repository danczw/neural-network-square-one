import math
import numpy as np

'''
What is a log? What does it do?
    - solving for x (natural log with base eulers number e):
    - e ** x = b
    - where b is input value
'''

b = 5.26

print(np.log(b))

# Check result - should equal b
print(math.e ** np.log(b))