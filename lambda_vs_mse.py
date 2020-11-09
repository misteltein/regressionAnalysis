import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data = np.loadtxt( 'boston.csv', skiprows = 23, delimiter = ',' );
y = data[ : , 13 ]
X = data[ : , : 13 ]
binWidth = 2

for j in range( X.shape[ 1 ] ):
    m = X[ : , j ].mean( )
    s = X[ : , j ].std( )
    X[ : , j ] = (X[ : , j ] - m ) / s

model = lm.Lasso( alpha = 0.023 )
model.fit( X , y )
print( model.intercept_ )
print( model.coef_ )
exit()

def region( k ):
    begin = binWidth * k
    end = binWidth * ( k + 1 )
    return begin, end

def ySplit( begin, end ):
    b = y[       : begin ]
    m = y[ begin : end   ]
    e = y[ end   :       ]
    return np.concatenate( ( b, e ) ), m

def XSplit( begin, end ):
    b = X[       : begin, : ]
    m = X[ begin : end  , : ]
    e = X[ end   :      , : ]
    return np.concatenate( ( b, e ) ), m

alphas = []
for z in range( -4, -1, 1 ):
    l = np.arange( 10**z, 10**(z+1) + 10**(z-1), 10**(z-1) );
    alphas = np.concatenate( ( alphas, l ) )

meanMSEs=[]
for a in alphas:
    mses = []
    for k in range( X.shape[0] // binWidth ):
        r = region( k )
        yT, yV = ySplit( *r )
        XT, XV = XSplit( *r )
        model = lm.Lasso( alpha = a )
        model.fit( XT, yT )
        b0   = model.intercept_
        beta = model.coef_
        yP = [ b0 + np.dot( beta, X_ ) for X_ in XV ]
        mse = mean_squared_error( yP, yV )
        mses.append( mse )
    meanMSE = np.mean( mses )
    meanMSEs.append( meanMSE )

fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel('$\\lambda$')
ax.set_ylabel('$\\langle {\\rm MSE}\\rangle$')
plt.xscale('log')
ax.plot(alphas,meanMSEs)
plt.show()
