#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install import-ipynb


# In[4]:


import pandas as pd
import import_ipynb
#from fbrefScraping import extractFromFBRef
import fbrefScraping as fbrscrap
import logging as log

#import requests as req
#from bs4 import BeautifulSoup
#import time
#import numpy as np
#import lxml
#import chardet
#import datetime

#from latest_fbref_url = 'https://fbref.com/en/comps/31/Liga-MX-Stats'


# In[ ]:





# In[4]:


def updateMatches(baseFile, newFile):    
    
    thisSeasonMatches = extractFromFBRef(latest_fbref_url)  
    baseDf = pd.read_csv(baseFile)
    
    updatedDf = pd.concat([baseDf,thisSeasonMatches], ignore_index= True)
    updatedDf = updatedDf.drop_duplicates(subset  = ["date","team_name"],    \
                                          keep    = 'last',                  \
                                          inplace = False)

    updatedDf.to_csv(newFile, index=False)
    log.info('Updated csv file called %s created successfully', newFile)

