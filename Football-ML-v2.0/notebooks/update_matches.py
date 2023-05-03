#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import import_ipynb
from fbrefScraping import extractFromFBRef
from fbrefScraping import loggingConfig

import fbrefScraping as fbrscrap
import logging as log


# In[ ]:


latest_fbref_url = 'https://fbref.com/en/comps/31/Liga-MX-Stats'


# In[ ]:


def updateMatches(baseFile, newFile):    
    
    thisSeasonMatches = extractFromFBRef(latest_fbref_url)  
    baseDf = pd.read_csv(baseFile, encoding = "latin-1")
    
    thisSeasonMatches["date"] = pd.to_datetime(thisSeasonMatches["date"])
    thisSeasonMatches["time"] = pd.to_datetime(thisSeasonMatches["time"])
    baseDf["date"] = pd.to_datetime(baseDf["date"])
    baseDf["time"] = pd.to_datetime(baseDf["time"])

    updatedDf = pd.concat([baseDf,thisSeasonMatches], ignore_index= True)
    updatedDf = updatedDf.drop_duplicates(subset  = ["date","team_name"],    \
                                          keep    = 'last',                  \
                                          inplace = False)

    updatedDf = updatedDf.replace('\u25c6','')
    updatedDf = updatedDf.sort_values(by = ["date", "time"], ascending = False)

    updatedDf.to_csv(newFile, index=False, encoding ="latin-1")
    log.info('Updated csv file called %s created successfully', newFile)
    


# In[ ]:


if __name__ == '__main__':
    loggingConfig(filename = 'updatingLogs.log', loglevel = 'DEBUG')
    log.info('Starting program...')
    print("Starting update sequence...")
    
    updateMatches("matches23-17-4.csv","matches23-17_updated.csv")
    
    print("Finishing update sequence...")
    
    log.info('Finishing program...')

    

