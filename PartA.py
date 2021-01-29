# -*- coding: utf-8 -*-
"""
Michael Ventoso
MichaelVentoso@Gmail.com

ProtAtOnce Interview Phase 3
Part A: NBA Dataset
"""
# Built-in
import os
import sys

# Libs
import numpy
import pandas
from scipy import stats
import matplotlib.pyplot

"""
Creates Pandas Dataframes for all given .csv files (given they are in the same folder)
Modifies player dataframe to match both statistics dataframes
Combines all three dataframes
Saves & returns combined dataframe
"""
def createCombinedDataFrame():

    #creates dataframe for first .csv file and sets column names
    nba_player_statistics_1 = pandas.read_csv(os.path.join(sys.path[0], 'NBA_player_statistics_1.csv'))

    nba_dataframe_1 = pandas.DataFrame(nba_player_statistics_1,
                                       columns=['X', 'Year', 'Player', 'Pos', 'Age', 'Tm', 'G', 'GS', 'MP', 'PER',
                                                'TS.', 'X3PAr',
                                                'FTr', 'ORB.', 'DRB.', 'TRB.', 'AST.', 'STL.', 'BLK.', 'TOV.', 'USG.',
                                                'blanl',
                                                'OWS', 'DWS', 'WS', 'WS.48', 'blank2', 'OBPM', 'DBPM', 'BPM', 'VORP',
                                                'FG', 'FGA',
                                                'FG.', 'X3P', 'X3PA', 'X3P.', 'X2P', 'X2PA', 'X2P.', 'eFG.', 'FT',
                                                'FTA', 'FT.',
                                                'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'])

    #creates dataframe for second .csv file and sets column names to match the first
    nba_player_statistics_2 = pandas.read_csv(os.path.join(sys.path[0], 'NBA_player_statistics_2.csv'))

    nba_dataframe_2 = pandas.DataFrame(nba_player_statistics_2,
                                       columns=['X', 'Year', 'Player', 'Pos', 'Age', 'Tm', 'G', 'GS', 'MP', 'PER',
                                                'TS.', 'X3PAr',
                                                'FTr', 'ORB.', 'DRB.', 'TRB.', 'AST.', 'STL.', 'BLK.', 'TOV.', 'USG.',
                                                'blanl',
                                                'OWS', 'DWS', 'WS', 'WS.48', 'blank2', 'OBPM', 'DBPM', 'BPM', 'VORP',
                                                'FG', 'FGA',
                                                'FG.', 'X3P', 'X3PA', 'X3P.', 'X2P', 'X2PA', 'X2P.', 'eFG.', 'FT',
                                                'FTA', 'FT.',
                                                'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'])

    #creates dataframe for thrid .csv file and sets column names
    nba_players = pandas.read_csv(os.path.join(sys.path[0], 'Players.csv'))

    nba_dataframe_players = pandas.DataFrame(nba_players,
                                             columns=['Player', 'height', 'weight', 'college', 'born',
                                                      'birth_city', 'birth_state'])

    #removes the X, and both blank columns from the first two dataframes
    nba_dataframe_1 = nba_dataframe_1.drop(['X', 'blanl', 'blank2'], axis=1)
    nba_dataframe_2 = nba_dataframe_2.drop(['X', 'blanl', 'blank2'], axis=1)

    #combines the first and second dataframes
    nba_dataframe_combo = nba_dataframe_1.append(nba_dataframe_2)

    #replaces spaces with underscores in third dataframe (so the 'Player' columns' strings' formats match
    nba_dataframe_players['Player'] = nba_dataframe_players['Player'].str.replace(" ", "_")

    #merges the players dataframe with the combined statistics frame
    nba_dataframe = pandas.merge(nba_dataframe_combo, nba_dataframe_players, on='Player')

    #saves combined dataframe to .csv
    nba_dataframe.to_csv("Combined_dataframe.csv");

    #return the combined dataframe
    return nba_dataframe


"""
Creates a histogram of the NBA players organized by age
The lowest and highest ages are 18 and 44 respectively.
Displays, and saves histogram as Age_histogram.png
"""
def ageHistogram(df):

    #get the relevant column from dataframe
    age_column = df['Age']

    #gets the min and max ages
    max = df.max(axis=0)['Age'].astype(int)
    min = df.min(axis=0)['Age'].astype(int)

    #uses min and max ages to calculate number of histogram bins (one per year)
    num_bins = max - min + 1

    #creates histogram and sets labels
    df.hist(column='Age', bins=num_bins, grid=False)
    matplotlib.pyplot.suptitle("NBA Players of a given age")
    matplotlib.pyplot.xlabel("Age")
    matplotlib.pyplot.ylabel("Number of NBA Players")

    #displays and saves histogram
    matplotlib.pyplot.savefig("Age_histogram.png")
    matplotlib.pyplot.show()


#Returns the average age of all players in given dataframe (rounded to two decimal places)
def averageAge(df):
    average_age = float("{0:.2f}".format(df.mean(axis=0)['Age']))
    return average_age


"""
Compares the average age of the players on each team to the average age of every NBA player
Prints:
    A. The average age of the NBA players as a whole
    B/C. The number of teams with an average player age lower/higher than the NBA average
Saves dictionary (team abbreviation & team's average age) to txt file
"""
def averageAgeOfTeams(df):

    #initializes dictionary
    age_dict = {}

    #calculates and print the average NBA player's age for comparison
    average_age = averageAge(df)
    print("The average NBA player is", average_age, "years old.")

    #iterates through dataframe
    for row in df.itertuples():

        #grabs team and age information from tuple
        team = row[5]
        age = row[4]

        #if there is no age value, skip this player
        if numpy.isnan(age):
            continue

        #adds player's age to running total for given team and increment teams player count by one
        #if team is not in dictionary already, add it
        if team in age_dict:
            age_dict.update({team: [age_dict[team][0] + age, age_dict[team][1] + 1]})
        else:
            age_dict.update({team: [age, 1]})

    #initializes counting variables
    num_teams_above = 0
    num_teams_below = 0

    #iterates through all the teams
    for team in age_dict:

        #divides team total age by number of players on that team
        team_average_age = age_dict[team][0]/age_dict[team][1]

        #determines if the team's average age is higher/lower than the NBA average
        if team_average_age >= average_age:
            num_teams_above += 1
        elif team_average_age < average_age:
            num_teams_below += 1

        #updates dictionary such that it now includes only k/v pairs of teams & their average player age
        age_dict.update({team: float("{0:.2f}".format(team_average_age))})

    #print raw data and line for spacing
    print("There are",num_teams_below, "teams with an average team age lower than the average age of all NBA players.")
    print("There are", num_teams_above, "teams with an average team age higher or equal to the average age of all NBA players.")
    print()

    #save dictionary with teams and average team age's to file, to avoid a large print to console
    with open('TeamAverageAges.txt', 'w') as team_average_file:
        print(age_dict, file=team_average_file)


"""
Creates scatterplot of the top shooters vs. their age
Plot includes a line to show the average NBA player's age
Top shooter is defined as the top 10 players with respect to field goal percentage
Top shooters also must have attempted at least a thousand field goals
This is to avoid players who only attempted/made one/a few shot(s) in their whole career (and thus have a FG% of 1.00)
A restriction I placed on myself was to do this task without iterating over the dataframes (or any data types for that matter)
"""
def ageOfTopShooters(df):

    #calculates average age for plot
    average_age = averageAge(df)

    #reduces large dataframe to include only the necessary columns
    simplified_df = df[['Age', 'FG.','FGA']].copy()

    #applies filters to dataframe to avoid any iteration
    filtered_df = simplified_df[simplified_df['FG.'] >= .5]
    double_filtered_df = filtered_df[filtered_df['FGA'] >= 1000]

    #selects only the top 10 shooters that meet the available criteria
    top_shooters = double_filtered_df.nlargest(10,'FG.')


    #creates scatterplot, and vertical line showing the average NBA player's age for reference
    matplotlib.pyplot.clf()
    top_shooters.plot.scatter(y='FG.', x='Age')
    matplotlib.pyplot.axvline(x=average_age)

    #sets labels for plot
    matplotlib.pyplot.suptitle("Top Shooters' FG% vs. Age")
    matplotlib.pyplot.xlabel("Age")
    matplotlib.pyplot.ylabel("Percentage of Field Goals Made")

    #displays and saves the scatterplot
    matplotlib.pyplot.savefig("TopShooters.png")
    matplotlib.pyplot.show()


"""
Since it seemed the top shooters from the previous figure were 
not all from the same age group, I predict:

Hypothesis: There will be no correlation between the age of the 
player and how well they are at shooting free throws.

This time, I want to analyze more than the top shooters, but in order to 
remove outliers I will limit the dataset to only players who have attempted
at least 100 field goals.

The statistical tests used in this function:
    Anderson-Darling Normality Test
    Pearson Correlation Test

This function:
Determines if the data to analyze is normally distributed
If not, prints error message and returns
If data is normally distributed, performs correlation test
Prints related values:
    r (correlation value)
    p (two-tailed p value)
Creates/saves scatterplot of data
"""
def ageToFGCorrelation(df):

    #reduces large dataframe to include only the necessary columns
    #filters out players who have attempted less than 100 fg attempts
    #also removes rows that have empty data for any of those columns
    simplified_df = df[['Age', 'FG.', 'FGA']].copy()
    filtered_df = simplified_df[simplified_df['FGA'] >= 100]
    filtered_df = filtered_df.dropna()

    #performs normality test
    anderson_results_age = stats.anderson(filtered_df['Age'], dist='norm')
    anderson_results_fg = stats.anderson(filtered_df['FG.'], dist='norm')

    #if data is not normal, prints error message and returns
    if anderson_results_age[0] < .05 or anderson_results_fg[0] < .05:
        print("The data received does not pass the Anderson-Darling test for normal distribution.")
        print("Because of this, we cannot perform further statistical tests on this data.")
        return

    #performs correlation test on data and prints relevant values
    pearson_results = stats.pearsonr(filtered_df['Age'], filtered_df['FG.'])
    print("The Pearson Correlation test shows an r-value of" , pearson_results[0])
    print("and a p-value of ", pearson_results[1],".")
    print()

    # creates scatterplot
    matplotlib.pyplot.clf()
    filtered_df.plot.scatter(y='FG.', x='Age')

    # sets labels for plot
    matplotlib.pyplot.suptitle("Age vs. % of Field Goals Made")
    matplotlib.pyplot.xlabel("Age")
    matplotlib.pyplot.ylabel("Percentage of Field Goals Made")

    # displays and saves the scatterplot
    matplotlib.pyplot.savefig("Age_fg_correlation.png")
    matplotlib.pyplot.show()


"""
Main simply calls the three data analysis and one hypothesis based functions
The started and finished prints are a personal preference, probably because I am used to so modeling
"""
if __name__ == '__main__':
    print("STARTED")
    print()
    nba_dataframe = createCombinedDataFrame()
    ageHistogram(nba_dataframe)
    averageAgeOfTeams(nba_dataframe)
    ageOfTopShooters(nba_dataframe)
    ageToFGCorrelation(nba_dataframe)
    print("FINISHED")
