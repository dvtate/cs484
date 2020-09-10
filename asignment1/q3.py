# imports
import pandas as pd
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import math
import functools

# Load data
df = pd.read_csv('Fraud.csv')

############
# Q3.a
############
print('Q3.a')

# Calculate fraud rate
fraud_n = functools.reduce(lambda a, v: a + v, df['FRAUD'], 0)
fraud_rate = fraud_n / len(df['FRAUD'])
print('\tfraudulent accounts:', fraud_n)
print('\ttotal accounts:', len(df['FRAUD']))
print("\tfraud rate: %3.4f%%" % (100 * fraud_rate))

############
# Q3.b
############
print('\nQ3.b\n\tsee charts')

# Box plot for each field
for field in df:
    # Skip external fields
    if field in ('CASE_ID', 'FRAUD'):
        continue

    # Draw boxplot
    df.boxplot(column=field, by='FRAUD', vert=False, whis=1.5)
    plt.title("Boxplot of %s by levels of FRAUD" % field)
    plt.xlabel(field)
    plt.ylabel('fraud')
    plt.grid(axis="y")
    plt.show()

############
# Q3.c
############
print('\nQ3.c')


# Orthonormalize using the orth function 
import scipy
from scipy import linalg as LA

mat = df[[field for field in df if field not in ('CASE_ID', 'FRAUD')]].to_numpy()
orthx = LA.orth(mat)
print(orthx)

# kek me

###