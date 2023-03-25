#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests as req
from bs4 import BeautifulSoup
import time
import numpy as np
import lxml
import chardet
import datetime
import logging as log

latest_fbref_url = 'https://fbref.com/en/comps/31/Liga-MX-Stats'


# In[2]:


def getSeasonLeagueTeamsUrls(ligaMxUrl):
    data = req.get(ligaMxUrl)
    soup = BeautifulSoup(data.text, "lxml")

    standings_table=soup.select('table.stats_table')[0]

    #Retrieving team links
    links = standings_table.find_all('a')
    links = [l.get("href") for l in links]
    links = [l for l in links if '/squads/' in l]

    league_urls = [f"https://fbref.com{l}" for l in links]
    return league_urls, soup


# In[3]:


def getMatches(team_url):
    team_web_data = req.get(team_url)
    team_matches = pd.read_html(team_web_data.text, match = "Scores & Fixtures")[0]
    return team_web_data, team_matches


# In[4]:


def getShooting(team_data):
    #soup = BeautifulSoup(team_data.text, features = "html.parser")
    soup = BeautifulSoup(team_data.text, 'lxml')

    shooting_link = soup.find_all("a")
    shooting_link = [l.get("href") for l in shooting_link]
    shooting_link = [l for l in shooting_link if l and 'all_comps/shooting/' in l]

    shooting_data = req.get(f"https://fbref.com{shooting_link[0]}")
    shootingInfo = pd.read_html(shooting_data.text, match='Shooting')[0]

    #transformating shooting info
    shootingInfo.columns = shootingInfo.columns.droplevel()
    
    shootingNewColumns = ["Date","Sh","SoT", "Dist","FK","PK","PKatt"]
    for newColumn in shootingNewColumns:
        if newColumn not in shootingInfo.columns:
            shootingInfo[newColumn] = np.nan

    return shootingInfo



# In[5]:


def extractFromFBRef(baseUrl, years=[]):
    
    if years == []:
        today = datetime.date.today()
        year = today.strftime("%Y")
        years = [year]
        
    all_matches = []
    
    for year in years:
        log.debug('Extracting season teams info from %s', baseUrl)
        league_urls, seasonSoup = getSeasonLeagueTeamsUrls(baseUrl)
        

        # Find anchor element with class 'prev' which contains link to previous season
        previous_season = seasonSoup.select("a.prev")[0].get("href")
        baseUrl = f"https://fbref.com{previous_season}"
        log.debug('Previous season link: %s', baseUrl)
        

        for team_url in league_urls:
            team_name = team_url.split("/")[-1].replace("-Stats","").replace("-"," ")
            
            log.debug('Retrieving info from team %s in season %s',team_name, year)
            teamWebData, teamMatchesDf = getMatches(team_url)
            
            log.debug('Retrieving shooting info from team %s in season %s',team_name, year)
            teamShootingDf = getShooting(teamWebData)
            
            try:
                log.debug('Merging datasets...')
                team_df = teamMatchesDf.merge(teamShootingDf[["Date","Sh","SoT", "Dist",'FK','PK',"PKatt"]], on = "Date")
            except ValueError:
                log.error('Something went wrong while merging FBref dataframes')
                continue

            team_df["team_name"] = team_name
            team_df["season"] = year
            all_matches.append(team_df)
            time.sleep(1)   

    league_matches = pd.concat(all_matches)
    league_matches.columns = [c.lower() for c in league_matches.columns] 
    
    league_matches['formation']  = league_matches['formation'].str.replace('\u25c6','')
    league_matches = league_matches.sort_values(by = ["date", "time"], ascending = False, ignore_index=True)

    
    log.info('FBRef data retrieved successfully')
    
    return league_matches

       


# In[6]:


"""
def updateMatches(baseFile, newFile):    
    
    thisSeasonMatches = extractFromFBRef(latest_fbref_url)  
    baseDf = pd.read_csv(baseFile, encoding = "latin-1")
    
    updatedDf = pd.concat([baseDf,thisSeasonMatches], ignore_index= True)
    updatedDf = updatedDf.drop_duplicates(subset  = ["date","team_name"],    \
                                          keep    = 'last',                  \
                                          inplace = False)

    updatedDf = updatedDf.replace('\u25c6','')
    updatedDf = updatedDf.sort_values(by = ["date", "time"], ascending = False)

    updatedDf.to_csv(newFile, index=False, encoding ="latin-1", ignore_index=True)
    log.info('Updated csv file called %s created successfully', newFile)
"""    


# In[7]:


def infoConfigLogging(filename):
    log.basicConfig(filename = filename,                               \
                    format='%(asctime)s - %(levelname)s: %(message)s', \
                    datefmt='%d-%b-%y %H:%M:%S',                       \
                    level = log.INFO,                                  \
                    filemode ='w')       
    
def debugConfigLogging(filename):
    log.basicConfig(filename = filename,                         \
                    format='%(asctime)s - %(levelname)s: %(message)s', \
                    datefmt='%d-%b-%y %H:%M:%S',                       \
                    level = log.DEBUG,                                 \
                    filemode ='w')       
    
def errorLoggingConfig(filename):
    log.basicConfig(filename = filename,                         \
                    format='%(asctime)s - %(levelname)s: %(message)s', \
                    datefmt='%d-%b-%y %H:%M:%S',                       \
                    level = log.ERROR,                                 \
                    filemode ='w')       


# In[8]:


def loggingConfig(filename = 'scraping.log', loglevel = 'INFO'):
    if loglevel.upper() == 'DEBUG':
        debugConfigLogging(filename)
    elif loglevel.upper() == 'ERROR':
        loggingConfigAsError(filename)
    else :
        infoConfigLogging(filename)


# In[9]:


def main():

    loggingConfig()
    log.info('Starting program...')
    print('Print to Console')
    
    updateMatches("matches23-17_2.csv","matches23-17_3.csv")
    
    log.info('Finishing program...')



# In[ ]:


if __name__ == '__main__':
    main()
        


# In[ ]:


#updateMatches("matches23-17-4.csv","matches23-17-4.csv")


# In[11]:


years = list(range(2023, 2017, -1))


# In[12]:


#league_matches = extractFromFBRef(latest_fbref_url,years)


# In[13]:


league_matches


# In[14]:


#league_matches.to_csv("matches23-17-4.csv", encoding = 'latin-1', index=False)


# In[ ]:




