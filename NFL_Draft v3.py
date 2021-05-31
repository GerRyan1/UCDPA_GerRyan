# -*- coding: utf-8 -*-
"""
Created on Thu May 20 22:59:19 2021

@author: Windows 10
"""

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import CSV file
nfl_draft = pd.read_csv("nfl_draft_1970-2021.csv")

# Check import and look at summary info
print(nfl_draft.head())
print(nfl_draft.shape)
print(nfl_draft.info())
print(nfl_draft.describe())

# Set a new multi level index
nfl_draft = nfl_draft.set_index(["year", "pick"])

# Check shape and labels
print(nfl_draft.head())
print(nfl_draft.shape)
print(nfl_draft.loc[[(2021, 1)]])

# Check for duplicated rows
print(nfl_draft.duplicated().sum())

# Check nulls
print(nfl_draft.isnull().sum())
print(nfl_draft["to"].isnull().sum() / len(nfl_draft))

# Create a new DF of only players with an age listed
nfl_age = nfl_draft.dropna(subset=["age"])
print(nfl_age.shape)

# Calculate median age and replace nulls in age column
avg_age = nfl_age["age"].median()
nfl_draft = nfl_draft.fillna({"age":avg_age})

# Create a dictionary of values to replace nulls
# Assume nulls in playing statistics columns mean a player never played in the NFL
# Replace null values in these columns with zero's
nfl_null_cols = nfl_draft.loc[:,"carAV":"tackles"].columns
null_dict = {i:0 for i in nfl_null_cols}
nfl_draft = nfl_draft.fillna(null_dict)

# Fill null colleges with 'unknown'
nfl_draft = nfl_draft.fillna({"college":"unknown"})
print(nfl_draft.isnull().sum())

# create a function for the top and bottom 10%
def pct10(column):
    return column.quantile(0.1)
def pct90(column):
    return column.quantile(0.9)
# Return the 10% and 90% percentile ages by position
print(nfl_age.groupby("position")["age"].agg([pct10,pct90]))

# Update "Team" column with recognised Team abreviations 
nfl_teams_old = list((nfl_draft.groupby("team")[["round"]].count()).index)
print(nfl_teams_old)
nfl_teams_modern = ["ARI","ATL","BAL","NE","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC","LAC","LAR","LVR","MIA","MIN","NO","NE","NYG","NYJ","LVR","PHI","ARI","PIT","LVR","LAR","LAC","SEA","SF","LAR","TB","TEN","WAS"]
nfl_draft["team"] = nfl_draft["team"].replace(nfl_teams_old, nfl_teams_modern)
print(nfl_draft.head())

# Load dataframe with NFL franchise information and prepare for merging
nfl_teams = pd.read_csv("nfl_teams.csv")
print(nfl_teams.head())

# Delete references to teams who have since changed name/location
nfl_teams = nfl_teams.dropna(subset=["team_division"])
nfl_teams = nfl_teams[(nfl_teams["team_name"] != "Washington Redskins") & (nfl_teams["team_name"] != "San Diego Chargers")]

# Join nfl_draft and nfl_teams to provide extra info on the drafting teams
nfl_draft = nfl_draft.reset_index()
nfl_merge = pd.merge(nfl_draft, nfl_teams, left_on="team", right_on="team_id", how="left", validate="many_to_one")
nfl_merge = nfl_merge.set_index(["year", "pick"])
print(nfl_merge.shape)
print(nfl_draft.shape)
nfl_merge = nfl_merge.drop(columns=["team_id_pfr", "team_conference_pre2002", "team_division_pre2002"])
print(nfl_merge.head())

# Extract the top 10 players with the most games and show their pro-bowl nominations
nfl_games = nfl_merge.sort_values("games", ascending=False)
nfl_games_top10 = nfl_games.iloc[0:10]
print(nfl_games_top10)
sns.barplot(data=nfl_games_top10.sort_values("pro_bowl"), x="pro_bowl", y="player", hue="position")
plt.title("Pro Bowl Appearances of NFL Players with Most Games")
plt.ylabel("Player")
plt.xlabel("Pro Bowl Appearances")
plt.show()

# Scatter plot of passing yards and passing TD's (Quarterbacks only)
nfl_qb = nfl_merge[nfl_merge["position"]=="QB"]
plt.scatter(nfl_qb["pass_yards"], nfl_qb["pass_tds"])
plt.title("Quarterbacks drafted since 1970: Passing Yards and Passing Touchdowns")
plt.ylabel("Passing Touchdowns")
plt.xlabel("Passing Yards")
plt.show

# lmplot of relationship between round selected and number of games played (Quarterbacks draft pre-2016 only)
nfl_qb2 = nfl_qb.loc[1970:2016]
sns.lmplot(data=nfl_qb2, x="round", y="games", x_estimator=np.mean)
plt.ylim(bottom=0)
plt.title("Quarterbacks drafted 1970-2016: Avg. Games Played based on Round Selected")
plt.ylabel("Games Played")
plt.xlabel("Round Selected")
plt.show()

# Which teams select the players that go on to have the most "All Pro" selections
nfl_grouped = nfl_merge.groupby("team_name").sum()
plt.barh(nfl_grouped.index[::-1], nfl_grouped["all_pro"][::-1])
plt.title("'All-Pro' Appearances by Team Drafting", color="blue")
plt.ylabel("Team", color="blue", fontsize=10)
plt.yticks(fontsize=7)
plt.xlabel("No. of All-Pro appearances by players drafted by this team", color="blue", fontsize=10)
plt.show()

# Which teams select the players that turn out to be the biggest 'flops'
nfl_2016 = nfl_merge.loc[:2016]
nfl_flop = nfl_2016[(nfl_2016["round"]<=2) & (nfl_2016["games"]<40)]
print(nfl_flop)
sns.countplot(data=nfl_flop.sort_values("team_name"), y="team_name")
plt.title("1st and 2nd Round Draft Pick 'Flops' by Team (1970-2016)", color="blue")
plt.ylabel("Team", color="blue", fontsize=10)
plt.yticks(fontsize=7)
plt.xlabel("Number of 1st and 2nd Round Picks with less than 40 games", color="blue", fontsize=10)
plt.show()


# Top 20 colleges who have provided the most first round draft picks since 1970
nfl_first_round = nfl_merge[nfl_merge["round"]==1]
nfl_colleges = nfl_first_round.groupby("college").count()
nfl_colleges = nfl_colleges.sort_values("round", ascending=False)
nfl_colleges_top20 = nfl_colleges.iloc[0:20]
sns.barplot(data = nfl_colleges_top20, x = "round", y=nfl_colleges_top20.index)
plt.title("Number of 1st Round Draft Picks by College (1970-2021)", color="blue", fontsize=20)
plt.ylabel("College", color="blue", fontsize=16)
plt.yticks(fontsize=12)
plt.xlabel("Number of 1st Round Picks", color="blue", fontsize=16)
plt.show()

# Compare number of first round draft picks provided by Alabama and USC since 1970
nfl_alabama = nfl_first_round[nfl_first_round["college"]=="Alabama"]
nfl_usc = nfl_first_round[nfl_first_round["college"]=="USC"]

fig, ax = plt.subplots()
ax.hist(nfl_alabama.reset_index()["year"], label="Alabama", alpha=0.4, bins=[1970,1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025])
ax.hist(nfl_usc.reset_index()["year"], label="USC", alpha=0.4, bins=[1970,1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025])
ax.set_title("No. of First Round Draft Selections by College", fontsize=16)
ax.set_xlabel("Year")
ax.set_ylabel("# of first round draft selections")
ax.legend()
plt.show()