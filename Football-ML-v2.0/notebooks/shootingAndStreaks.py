import pandas as pd
import requests as req
from bs4 import BeautifulSoup
import time
import numpy as np
import lxml
import chardet
import logging as log

fbRefFileName = 'matches23-17_updated.csv'


# create a function to convert string values to numerical values
def result_to_numeric(result):
    if result.lower() in ["win","w"]:
        return 1
    elif result.lower() in ["loss","l"]:
        return -1
    elif result.lower() in ["draw","d"]:
        return 0



#Dataframe Initial Transformations 

#Normalizing team and opponent names
class MissingDict(dict):
    #if name not included in following dict, keep the name the same instead of deleting
    __missing__ = lambda self, key:key

df = pd.read_csv(fbRefFileName, encoding = 'latin-1')
df["result_num"] = df["result"].apply(result_to_numeric)
df["ga"] = pd.to_numeric(df["ga"], errors = 'coerce').convert_dtypes()
df["gf"] = pd.to_numeric(df["gf"], errors = 'coerce').convert_dtypes()
df = df.sort_values(by = ["date", "time"], ascending = True, ignore_index=True)
thisDf = df.copy(deep = True)

   
map_values = {
    "América": "America",
    "Atlético": "Atletico",
    "FC Juárez":"FC Juarez",
    "León":"Leon",
    "Mazatlán":"Mazatlan",
    "Querétaro":"Queretaro",
    "Santos Laguna":"Santos"
}  

mapping = MissingDict(**map_values)
thisDf["team_new"] = thisDf["team_name"].map(mapping)
thisDf["opponent_new"] = thisDf["opponent"].map(mapping)
#thisDf




#Calculating average shooting info for each team
def rollingAverages(teamDf, cols, newCols):
    teamDf = teamDf.sort_values("date")
    rolling_avgs=teamDf[cols].rolling(window=4, closed = 'left').mean()
    teamDf[newCols] = rolling_avgs
    
    #Drop rows when there are not enough previous matches info (at the beginning of the dataset)
    teamDf = teamDf.dropna(subset = newCols)

    return teamDf

#Implementation
cols = ["gf","ga","sh","sot","dist","pk","pkatt"]
newCols = [f"{c}_avg" for c in cols]

matches_rolling = thisDf.groupby("team_name").apply(lambda x: rollingAverages(x, cols, newCols))
matches_rolling = matches_rolling.droplevel('team_name')
matches_rolling.index = range(matches_rolling.shape[0])



#Streak calculator
def calc_streaks1(series, streakType):
    current_streak = 0
    streaks=[]
    
    if streakType.lower() in ['unbeaten','u']:
        numMatchResult = [ result_to_numeric('w'), result_to_numeric('d') ]
    else:
        numMatchResult = [ result_to_numeric(streakType.lower()) ]
    
    for actualResult in series:
        if actualResult not in numMatchResult:
            current_streak =  0
        else:
            current_streak += 1

        streaks.append(current_streak)

    return streaks