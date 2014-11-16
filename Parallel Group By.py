
# coding: utf-8

# In[1]:

from IPython.parallel import Client
import json
rc = Client()
rc.ids
print( 'We have %s clients'%len(rc.ids ))
dview = rc[:]


# In[2]:

import pandas as pd
import numpy as np

import numpy as np
import pandas as pd
print( 'Hello World' )

N = 37171567
m  = 118171

A = np.random.random_integers( 0, m, N )
B = np.random.random_integers( 0, m, N )
#D = [ chr( 98+x ) for x in np.random.random_integers( 0, 25, N ) ]
C = np.random.randn( N )
#df = pd.DataFrame( {'A':A, 'B':B, 'C':C, 'D':D } )
df = pd.DataFrame( {'A':A, 'B':B, 'C':C } )
print( df.head(10) )


# In[3]:

def print_attributes( obj ):
    print( '\n'.join( x for x in dir( obj ) if not x.startswith('_' ) ) )


# In[4]:

def split_frame( frame, key, num_splits ):
    """ Split frame into (very roughly) approximately equal sizes, by key, using key
    """
    
    percentiles = np.linspace( 0.0, 100.0, num_splits+1 )
    percentile_bounds = np.percentile( df[ key ].values, list(percentiles ) )
    
    taken = pd.Series( False, index=df.index )
    frames = []
    for i in range( num_splits ):
        ub = percentile_bounds[i+1]
        flt = ( ~taken ) & ( frame[ key ]<=ub )
        taken = taken | flt
        frames.append( df[ flt ] )
    assert np.all( taken )
    return frames

#print( split_frame( df, 'A', 3 ) )


# In[5]:

def p_assign_frame( rc, frame, client_variable_name, key ):
    """ A split a dataframe, and assign it across the clients
    rc - IPython.parallel.Client()
    frame - a dataframe
    client_variable_name - name of the frame's variable
    key - the column that we use for the splitting.
    """    
    num_splits = len( rc.ids )
    
    frames = split_frame( frame, key, num_splits )
    
    for i in range( num_splits ):
        rc[i][ client_variable_name ]=frames[i]
        
p_assign_frame( rc, df, 'df', 'A' )


# In[6]:

def p_group_by( rc, client_group_name, client_frame_name, *args ):
    q = r"""%s=%s.groupby(%s)"""%( client_group_name, client_frame_name, json.dumps( args ) )
    rc[:].execute( q )

p_group_by( rc, 'grp', 'df', 'A' )       


# In[7]:

def p_group_by_dot( rc, client_group_name, q ):
    """ execute code against the client group
    """
    q = r"""p_group_by_dot_result=%s.%s"""%( client_group_name, q )
    #print( 'Executing %s'%q)
    rc[:].execute( q )
    df_result = pd.concat( rc[:][ 'p_group_by_dot_result' ] )
    rc[:][ 'p_group_by_dot_result' ] = None # delete the result
    return df_result

print( p_group_by_dot (rc, 'grp', 'sum()' ).head() )


# In[8]:

grp = df.groupby( 'A' )
print( grp.sum().head() )


# In[9]:

psum = p_group_by_dot (rc, 'grp', 'sum()' )
asum = grp.sum()

assert np.all( psum==asum )
print( 'Parallel and Serial results are the same' )


# In[10]:

from datetime import datetime
t0 = datetime.now()
p_assign_frame( rc, df, 'df', 'A' )
t1 = datetime.now()
print( t1-t0 )


# In[11]:


get_ipython().magic("timeit p_group_by( rc, 'grp', 'df', 'A' )")


# In[12]:

get_ipython().magic("timeit grp = df.groupby( 'A' )")


# In[14]:

# Comparing times...
get_ipython().magic("timeit p_group_by_dot (rc, 'grp', 'sum()' )")
get_ipython().magic('timeit grp.sum()')


# In[13]:



