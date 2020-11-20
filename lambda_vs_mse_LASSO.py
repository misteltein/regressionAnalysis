import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data = np.loadtxt( 'boston.csv', skiprows = 23, delimiter = ',' );
y = data[ : , 13 ]
X = data[ : , : 13 ]
binWidth = 46

for j in range( X.shape[ 1 ] ):
    m = X[ : , j ].mean( )
    s = X[ : , j ].std( )
    X[ : , j ] = (X[ : , j ] - m ) / s

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

alpha_begin = 1.0e-4
alpha_end   = 1.0e+2
numDiv      = 1000
ratio = np.power( alpha_end / alpha_begin, 1.0 / ( numDiv - 1 ) )
alpha_current = alpha_begin
alphas = [ alpha_begin ]
for i in range( numDiv ):
   alphas.append( alphas[ i ] * ratio ) 

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
#plt.yscale('log')
ax.plot( alphas, meanMSEs )
plt.title("LASSO")
plt.grid( which = 'major', color = 'gray', linestyle = 'dashed' )
#plt.show()
plt.savefig( "lambda_vs_mse_LASSO.png", format = "png", dpi = 600 )
