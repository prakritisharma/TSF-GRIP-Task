#!/usr/bin/env python
# coding: utf-8

# ## Task 5 for GRIP - TSF
# 
# ### Exploratory Data Analysis - Sports
# 
# ● Perform ‘Exploratory Data Analysis’ on dataset ‘Indian Premier League’ 
# 
# ● As a sports analysts, find out the most successful teams, players and factors contributing win or loss of a team. 
# 
# ● Suggest teams or players a company should endorse for its products. 
# 
# ● You can choose any of the tool of your choice
# 
# ### Done By
# >Prakriti Sharma for GRIPMAY21
# 
# ##### Dataset link: https://bit.ly/34SRn3b

# ### Importing Libraries

# In[378]:


#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn import metrics

import time
import matplotlib
import matplotlib.cm as cm
import altair as alt
import os
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import tkinter as tk
from PIL import ImageTk
import PIL.Image
import gc
import re

from wordcloud import WordCloud


# In[355]:


mdf = pd.read_csv('matches.csv')
mdf.head()


# In[356]:


ddf = pd.read_csv('deliveries.csv')
ddf.head()


# ### Data Cleaning

# In[357]:


#Data Cleaning
ddf.dropna(axis=0, how='any',inplace=True)
mdf.dropna(axis=0, how='any',inplace=True)


# ### Exploratory Data Analysis

# In[358]:


ddf.info()


# In[359]:


mdf.info()


# In[360]:


#plt.figure(figsize = (16, 10))
#sns.heatmap(ddf.corr(), annot = True, cmap="YlGnBu")
#plt.show()


# Here, we can see that features 'bye_runs' and 'penalty_runs' have no correlation with any of the features (hence, the discoloured area). So we delete these two features in the next statement.

# In[361]:


ddf = ddf.drop(['bye_runs','penalty_runs'], axis = 1)


# ##### Since both the dataframes have a common column for match_id, we merge both dataframes on that column.

# In[362]:


#merging both the dataframes
df = pd.merge(mdf, ddf, left_on='id', right_on='match_id')


# In[363]:


#finding which player won the 'Player of Match' title, how many times.
pom = pd.DataFrame(mdf.player_of_match.value_counts())


# In[364]:


#checking for null values
df.isnull().any()


# ##### Checking for categorical data variables, for further analysis.

# In[365]:


mdf.season.value_counts()


# In[366]:


mdf.result.value_counts()


# In[367]:


mdf.dl_applied.value_counts()


# In[368]:


mdf.toss_winner.value_counts()


# In[369]:


mdf.toss_decision.value_counts()


# In[370]:


mdf['toss'] = mdf['toss_winner']==mdf['winner']
mdf.toss.value_counts()


# ## Most Successful Teams

# In[371]:


###most successful teams
def team():
    plt.figure(figsize=(10,10))
    plt.xticks([5,8,11,14,17,21])
    ax = sns.countplot(y='winner', data=mdf, orient='h', palette = 'plasma',order=mdf.winner.value_counts().index)
    plt.xlabel('Wins', weight = 'bold', fontsize = 13)
    plt.title ("Number of matches won Team-wise", fontsize = 15, weight = 'bold')
    plt.ylabel('Teams', weight = 'bold', fontsize = 13)
    mng = plt.get_current_fig_manager()
    plt.show(block=False)
    mng.full_screen_toggle()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()


# ## Best Players

# In[372]:


### best players
pom = pom.loc[pom['player_of_match'] > 2]
def play():    
    x = list(pom.index)
    y = list(pom['player_of_match'])

    plt.figure(figsize = (16,5))
    sns.barplot(x,y, palette = 'rainbow')
    plt.ylabel("No of Player of Matches won", weight = 'bold', fontsize = 13)
    plt.xlabel("Players", weight = 'bold', fontsize = 13)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()


# ## Factors contributing to win

# In[373]:


def fact():
    #factor1
    plt.figure(figsize = (16, 10))
    sns.heatmap(ddf.corr(), annot = True, cmap="Blues")
    plt.title("Correlation of all features in deliveries.csv", weight = 'bold', fontsize = 16)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(20)
    plt.close()
    
    ###factors 2
    plt.figure(figsize=(6,6))
    space = [0.1,0.12]
    mdf['toss'].value_counts().plot(kind='pie', explode=space, fontsize=10,autopct='%1.1f%%', colors = ['skyblue', 'salmon'])
    plt.ylabel('')
    plt.title('Team that Won the Toss wins the match', weight = 'bold', fontsize=15)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    
    ###factor 3
    x = mdf.loc[mdf['toss_decision'] == 'field']
    plt.figure(figsize=(6,6))
    space = [0.1,0.12]
    x['toss'].value_counts().plot(kind='pie', explode=space, fontsize=10,autopct='%1.1f%%', colors = ['purple', 'yellow'])
    plt.ylabel('')
    plt.title('Team that Won the Toss and chose to field, won', weight = 'bold', fontsize=15)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    
    ###factor 4
    x = mdf.loc[mdf['toss_decision'] == 'bat']
    plt.figure(figsize=(6,6))
    space = [0.1,0.12]
    x['toss'].value_counts().plot(kind='pie', explode=space, fontsize=10,autopct='%1.1f%%', colors = ['cyan', 'red'])
    plt.ylabel('')
    plt.title('Team that Won the Toss and chose to bat, won', weight = 'bold', fontsize=15)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    
    ###factor 5
    plt.figure(figsize=(10,6))
    mdf['city'].value_counts().plot(kind='bar', fontsize=10, cmap = 'Set2')
    plt.xticks(rotation = 15)
    plt.xlabel('')
    plt.ylabel('Total Wins', weight = 'bold')
    plt.title('Depending on cities the matches were played in', weight = 'bold', fontsize=15)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()


# ### Team Analysis

# In[374]:


def study():
    ###analysis1
    plt.figure(figsize=(10,6))
    plt.bar(df['over'], df['batsman_runs'], color = 'maroon')
    plt.xticks(rotation = 15)
    plt.ylabel('Batsman runs')
    plt.title('Overs vs Batsman Runs', weight = 'bold', fontsize=15)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    
    ###analysis2
    plt.figure(figsize=(16,6))
    plt.bar(df['batting_team'], df['batsman_runs'], color = 'deepskyblue')
    plt.xticks(rotation = 15)
    plt.ylabel('Batsman runs')
    plt.title('Batting Team vs Batsman Runs', weight = 'bold', fontsize=15)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    
    ###analysis3
    plt.figure(figsize=(16,6))
    plt.bar(df['bowling_team'], df['player_dismissed'].count(), color = 'darkgreen')
    plt.xticks(rotation = 15)
    plt.ylabel('Players Dismissed')
    plt.yticks([800,900,1000,1100,1200])
    plt.title('Bowling Team vs Player Dismissed', weight = 'bold', fontsize=15)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    
    ###analysis4
    plt.figure(figsize=(16,6))
    plt.bar(df['bowling_team'], df['batsman_runs'], color = 'blueviolet')
    plt.xticks(rotation = 15)
    plt.ylabel('Batsman runs')
    plt.title('Bowling Team vs Batsman Runs', weight = 'bold', fontsize=15)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()


# ### Designing the Dashboard

# In[380]:


def center(win):
    """
    centers a tkinter window
    :param win: the root or Toplevel window to center
    """
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()

path=""
lang=""

def execute():
    global path
    global lang
    #global data
    root.geometry("500x450")
    root.title("IPL Data Dashboard")
    root.configure(background="paleturquoise")
    root.geometry("+400+115")
    
    lab = Label(root, text = "Welcome to IPL", font = ('Veranda',20, 'bold'), borderwidth = 2,bg="paleturquoise", fg = "darkviolet").grid(row = 0, column = 0, padx = 10, pady = 5)
    lab1 = Label(root, text = "Data Dashboard!", font = ('Veranda',20, 'bold'), borderwidth = 2,bg="paleturquoise", fg = "darkviolet").grid(row = 0, column = 1, padx = 10, pady = 5)

    #Button for PLOT1
    fp1 = open("plot4.png","rb")
    img1=PIL.Image.open(fp1)
    img1 = img1.resize((120,120))
    imgbtn1 = ImageTk.PhotoImage(img1)
    lab1 = Label(root, text = 'Best Team', bg = "coral", font = ('Veranda', 9), borderwidth = 1).grid(row = 1, column = 0)
    b1 = Button(root, image = imgbtn1, compound = TOP, bg = "black",command=team)
    b1.image = imgbtn1
    b1.grid(row=2,column=0, pady = 20, padx = 20)

    #Button for PLOT2
    fp2 = open(r"plot8.png","rb")
    img2=PIL.Image.open(fp2)
    img2 = img2.resize((120,120))
    imgbtn2 = ImageTk.PhotoImage(img2)
    lab2 = Label(root, text = 'Best Players', bg = "coral", font = ('Veranda', 9), borderwidth = 1).grid(row = 1, column = 1)
    b2 = Button(root, image = imgbtn2, compound = TOP, bg = "black",command=play)
    b2.image = imgbtn2
    b2.grid(row=2,column=1, pady = 20, padx = 20)

    #Button for PLOT3
    fp4= open(r"plot5.png","rb")
    img4=PIL.Image.open(fp4)
    img4 = img4.resize((120,120))
    imgbtn4 = ImageTk.PhotoImage(img4)
    lab4 = Label(root, text = "Factors for win", bg = "coral", font = ('Veranda', 9), borderwidth = 1).grid(row = 3, column = 0)
    b4 = Button(root, image = imgbtn4, compound = TOP, bg = "black",command=fact)
    b4.image = imgbtn4
    b4.grid(row=4,column=0, pady = 20, padx = 20)

    #Button for PLOT4
    fp3= open(r"plot6_1.png","rb")
    img3=PIL.Image.open(fp3)
    img3 = img3.resize((120,120))
    imgbtn3 = ImageTk.PhotoImage(img3)
    lab3 = Label(root, text = 'Team Analysis', bg = "coral", font = ('Veranda', 9), borderwidth = 1, padx = 10).grid(row = 3, column = 1)
    b3 = Button(root, image = imgbtn3, compound = TOP, bg = "black",command=study)
    b3.image = imgbtn3
    b3.grid(row=4,column=1, pady = 20, padx = 20)

from tkinter import Tk, Button, Entry, Label
root = Tk()
execute()
root.mainloop()

