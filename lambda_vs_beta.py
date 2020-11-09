import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

data = np.loadtxt( 'boston.csv', skiprows = 23, delimiter=',' );
y = data[ : , 13 ]
X = data[ : , : 13 ]

for j in range( X.shape[ 1 ] ):
    m = X[ : , j ].mean( )
    s = X[ : , j ].std( )
    X[ : , j ] = (X[ : , j ] - m ) / s

alphas = np.arange( 0.0001, 10.0, 0.0001 )
beta=[]
for a in alphas:
    model = lm.Lasso( alpha = a )
    model.fit( X , y )
    beta.append( model.coef_ )

plt.rcParams["legend.framealpha"] = 1
plt.rcParams['xtick.direction']   = 'in'
plt.rcParams['ytick.direction']   = 'in'

fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel('$\\lambda$')
plt.xscale('log')
for j in range( X.shape[1] ):
    l = '$\\beta_{' + str( j + 1 ) + '}$'
    ax.plot( alphas, [ b[ j ] for b in beta ], label = l )
ax.legend()
plt.show()
