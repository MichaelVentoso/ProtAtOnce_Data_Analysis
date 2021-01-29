# Michael Ventoso
# MichaelVentoso@Gmail.com
# 732.697.5992
#
# ProtAtOnce Phase 3
# Part A: NBA Dataset
# Dealing with Age

import numpy
import pandas
import matplotlib.pyplot


def createCombinedDataFrame():
    nba_player_statistics_1 = pandas.read_csv(
        r"/home/michaelventoso/PycharmProjects/ProtAtOnce_MVentoso/NBA_player_statistics_1.csv")

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

    nba_player_statistics_2 = pandas.read_csv(
        r"/home/michaelventoso/PycharmProjects/ProtAtOnce_MVentoso/NBA_player_statistics_2.csv")

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

    nba_dataframe_1 = nba_dataframe_1.drop(['X', 'blanl', 'blank2'], axis=1)
    nba_dataframe_2 = nba_dataframe_2.drop(['X', 'blanl', 'blank2'], axis=1)

    nba_dataframe_combo = nba_dataframe_1.append(nba_dataframe_2)

    nba_players = pandas.read_csv(
        r"/home/michaelventoso/PycharmProjects/ProtAtOnce_MVentoso/Players.csv")

    nba_dataframe_players = pandas.DataFrame(nba_players,
                                             columns=['Player', 'height', 'weight', 'college', 'born',
                                                      'birth_city', 'birth_state'])

    nba_dataframe_players['Player'] = nba_dataframe_players['Player'].str.replace(" ", "_")

    nba_dataframe = pandas.merge(nba_dataframe_combo, nba_dataframe_players, on='Player')

    # nba_dataframe.to_csv("combined.csv");

    # print(nba_dataframe.dtypes)
    return nba_dataframe


def ageHistogram(df):

    age_column = df['Age']

    max = df.max(axis=0)['Age'].astype(int)
    min = df.min(axis=0)['Age'].astype(int)

    num_bins = max - min + 1

    df.hist(column='Age', bins=num_bins, grid=False)

    matplotlib.pyplot.suptitle("NBA Players of a given age")
    matplotlib.pyplot.xlabel("Age")
    matplotlib.pyplot.ylabel("Number of NBA Players")
    matplotlib.pyplot.savefig("AgeHistogram.png")
    matplotlib.pyplot.show()


def averageAge(df):
    average_age = float("{0:.2f}".format(df.mean(axis=0)['Age']))
    return average_age


def averageAgeOfTeams(df):
    age_dict = {}

    average_age = averageAge(df)
    print("The average NBA player is", average_age, "years old.")

    for row in df.itertuples():
        team = row[5]
        age = row[4]

        if numpy.isnan(age):
            continue

        if team in age_dict:
            age_dict.update({team: [age_dict[team][0] + age, age_dict[team][1] + 1]})
        else:
            age_dict.update({team: [age, 1]})

    num_teams_above = 0
    num_teams_below = 0

    for team in age_dict:
        team_average_age = age_dict[team][0]/age_dict[team][1]

        if team_average_age >= average_age:
            num_teams_above += 1
        elif team_average_age < average_age:
            num_teams_below += 1

        age_dict.update({team: float("{0:.2f}".format(team_average_age))})

    print("There are",num_teams_below, "teams with an average team age lower than the average age of all NBA players.")
    print("There are", num_teams_above, "teams with an average team age higher than the average age of all NBA players.")
    print()

    with open('TeamAverageAges.txt', 'w') as f:
        print(age_dict, file=f)


def ageOfGoodShooters(df):
    average_age = averageAge(df)

    simplified_df = df[['Age', 'FG.','FGA']].copy()
    filtered_df = simplified_df[simplified_df['FG.'] >= .5]
    double_filtered_df = filtered_df[filtered_df['FGA'] >= 1000]

    age_total = double_filtered_df['Age'].sum()
    average_shooter_age = age_total / double_filtered_df.shape[0]

    top_shooters = double_filtered_df.nlargest(10,'FG.')

    top_shooters.plot.scatter(y='FG.', x='Age')
    matplotlib.pyplot.axvline(x=average_age)

    matplotlib.pyplot.suptitle("Top Shooters' FG% vs. Age")
    matplotlib.pyplot.xlabel("Age")
    matplotlib.pyplot.ylabel("Percentage of Field Goals Made")
    matplotlib.pyplot.savefig("TopShooters.png")
    matplotlib.pyplot.show()


if __name__ == '__main__':
    print("STARTED")
    print()
    nba_dataframe = createCombinedDataFrame()
    ageHistogram(nba_dataframe)
    averageAgeOfTeams(nba_dataframe)
    ageOfGoodShooters(nba_dataframe)
    print("FINISHED")
