import numpy as np
import sklearn.linear_model as lm

data = np.loadtxt( 'boston.csv', skiprows = 23, delimiter = ',' );
y = data[ : , 13 ]
X = data[ : , : 13 ]

for j in range( X.shape[ 1 ] ):
    m = X[ : , j ].mean( )
    s = X[ : , j ].std( )
    X[ : , j ] = (X[ : , j ] - m ) / s

model = lm.Ridge( alpha = 0.001 )
model.fit( X , y )
print( model.intercept_ )
print( model.coef_ )
