# About

I have some vague ideas about buildinga model to predict spreads (which could inform betting - bet when model's spread differs significantly from book spread). Not exactly sure which features to use but im curious about advanced stats like EPA and success rate. Or perhaps even just win pct, average team pass yards offense/defense, team rush yards offense/defense, team points offense/defense, net turnovers, etc. For both home and away team. So a vector kinda like this (but ultimately not in json format, just did that for readability):

```json
{
  "home": {
    "win_pct": 0.7,
    "games_played": 10,
    "avg_pass_yards_offense": 200,
    "avg_pass_yards_defense": 200,
    "avg_rush_yards_offense": 100,
    "avg_rush_yards_defense": 100,
    "avg_points_offense": 20,
    "avg_points_defense": 20,
    "avg_EPA": 0.12,
    "avg_success_rate": 0.5,
    "actual_spread": 7
  },
  "away": {
    "win_pct": 0.5,
    "games_played": 10,
    "avg_pass_yards_offense": 200,
    "avg_pass_yards_defense": 200,
    "avg_rush_yards_offense": 100,
    "avg_rush_yards_defense": 100,
    "avg_points_offense": 20,
    "avg_points_defense": 20,
    "avg_EPA": 0.12,
    "avg_success_rate": 0.5,
    "actual_spread": 7
  }
}
```

Notes:

- win pct wouldnt really work for early games in the season. kinda true of the other stats as well? small sample size early on (or is each game basically a sample of 1 actually?).
- I dont think I need the predicted spread for the training
- could perhaps combine home pass offense and away pass defense into one feature (and same for other stats) to bake relationship in?
- games played/win pct could probably be used to derive record with complete accuracy (not sure that would even be necessary though)

Then I would train the model on all the games I can on a game-by-game basis. So for week 1 of season 2019, it would look at all the features for each game to figure out how they contribute to the spread. And so on for all seasons except 1 or 2 for validation purposes. Could compare against average difference between actual and predicted vegas spreads (if I can find them).

# Resources

- nflfastR: https://github.com/nflverse/nflfastR
  - client for pbp nfl data. think it ultimately comes from the nfl game books.
- nflfastR data (pbp and other): https://github.com/nflverse/nflverse-data/releases
- nflfastR data field descriptions: https://www.nflfastr.com/articles/field_descriptions.html
- getting nflfastR data into python: https://gist.github.com/Deryck97/dff8d33e9f841568201a2a0d5519ac5e
- nfl game books: https://support.nfl.com/hc/en-us/articles/4989073286044-Game-Books
  - official nfl game stats published as pdf. should be able to find on the nfl game summary page such as https://www.nfl.com/games/texans-at-ravens-2023-post-2?active-tab=watch
- reddit discussion about how to aggregate pass yards from game from pbp data (using R lib): https://www.reddit.com/r/NFLstatheads/comments/yq3wan/team_passing_yards_nflfastr/
- article on accuracy of vegas odds: https://www.theonlycolors.com/2020/9/29/21492301/vegas-always-knows-a-mathematical-deep-dive
  - could be useful for comparing the accuracy of my model. in particular "Distribution of the deviation of the final margin of victory from the Vegas spread"
  - for example, perhaps avg spread difference between vegas and reality is ~10 so a model with an average difference of 8 would be good

# TODO:

- [x] setup venv, install dependencies, handle dependencies (poetry? requirements.txt?)
  - using poetry. love it compared to alternatives so far.
- [x] script to get data from github repo
- [x] script to load into pd data frame
- [x] identify required stats for predicting spread. start simple.
  - rush/pass yards offense/defense, points scored/allowed, actual spread
- [x] setup workflow for jupyter notebook
  - [x] install ipykernel and use venv/kernal in notebook. can use this get venv location `poetry show -v`
  - [x] refactor data get/load into functions i can import.
    - path error... relative paths not resolving when importing
- [ ] aggregate game stats from play by play for each game, for each season. keep the features simple for now.
  - [x] not sure if this should be dataframe or sqlite table maybe pandas but also sqlite for dev purposes (if still not using notebook).
    - dataframe for sure to make it easier to develop.
  - [ ] figure out how to model it
    - Season dataframe with each row being a game. WeekNumber, Team1, Team2, Team1Score, Team2Score, Team1PassYards, Team2PassYards, etc.
      - could be Home and Away team instead of 1,2
      - not sure if squashing to 1 row is the best way to handle teams? Could have seperate row for each team per game (so two rows per game, 1 for each team)
        - Say I want to get average team stats through week 10 for the bears. I have to group by either column at the same time? maybe difficult/impossible?
        - I can always smash them into 1 record later for the training input too...
      - I think season is too aggregated... i need individual games so i can accumulate them through a given number of weeks for any given season.
    - Games dataframe with:
      - Team, pass off, pass def, rush off, rush def, average EPA
- [ ] use aggregated team stats to get average stats for each team coming into each game. this will be
  - should be able to use to query something like `"SELECT AVERAGE(pass_yards) as avg_pass_yards, etc. FROM Season where Team1 = 'CHI' or Team2 = 'CHI' and week_number < 10"`
    - this would be the input to the model to predict the week 10 game for the chicago bears
    - this is going to be a lot of duplicate aggregation if I do this for each week. may be a good way to use previously computed averages. maybe storing totals and counts for each stat in a separate table and then computing averages from that table.
- [ ] cleanup
  - [ ] the get_data and load_data is duplicated in data.py and get_data.py/load_data.py. just use one or the other.
  - [ ] move notebook code to python files. think about a managable way to share logic between notebook and python files so I can drop into the pipeline and inspect as needed.
    - probably just put everything in a functions that are imported into the python file and notebook?
- [ ] simple model to predict spread
  - [ ] use sklearn to train model (linear regression, although maybe not ideal because relationship between features (pass off vs. opponent pass def))
  - [ ] some sort of basic analysis to see how it performed. including manually comparing to vegas spread (maybe I can find an average difference? https://www.theonlycolors.com/2020/9/29/21492301/vegas-always-knows-a-mathematical-deep-dive)
- [ ] improve features. either at game aggregation level or team @ week aggregation level
  - W/L record or games played and win pct?
  - success rate (calculate success (0 or 1) from each play).
    - could be measured from positive EPA. or 40% of yards to go on 1st, 70% on 2nd, 100% on 3rd/4th (or similar)
  - total points scored/allowed

# Current status:

- project managed by poetry
- Can fetch all play by play data in compressed csv format from ` nflfastR`` releases using  `get_data.py``
- Can load raw data into pandas dataframe and sqlite3 using load_data.py. data types are inferred by pandas and sqlite is created from pandas method. this is key because there are a lot of columns (372 at time of writing). When identifying actual columns I want to use I can get more specific with data types.

# Modeling ideas

Perhaps somethinglike this for aggregating stats

Bears pass for 170 week 1, and 220 week 2. pass_off_running_avg is the running average, so their average pass yards/game coming into week 3 is 195. This would be the input to the model for predicting the week 3 game along with the bookie spread (and other team).

| season | week | team | pass_off_running_avg | etc... |
| ------ | ---- | ---- | -------------------- | ------ |
| 2023   | 1    | DET  | null                 |        |
| 2023   | 1    | CHI  | null                 |        |
| 2023   | 2    | DET  | 260                  |        |
| 2023   | 2    | CHI  | 170                  |        |
| 2023   | 3    | DET  | 276                  |        |
| 2023   | 3    | CHI  | 195                  |        |
| etc... |      |      |                      |        |

Then I could do SELECT pass_off_running_avg FROM Season where Team = 'CHI' week = 3

# possible spread -> win pct map

from image here: https://www.theonlycolors.com/2020/9/29/21492301/vegas-always-knows-a-mathematical-deep-dive

Spread,Victory%
0.5,51.4%
1.0,52.8%
1.5,54.2%
2.0,55.6%
2.5,57.0%
3.0,58.4%
3.5,59.8%
4.0,61.2%
4.5,62.5%
5.0,63.8%
5.5,65.2%
6.0,66.5%
6.5,67.7%
7.0,69.0%
7.5,70.2%
8.0,71.5%
8.5,72.7%
9.0,73.8%
9.5,75.0%
10.0,76.1%
10.5,77.2%
11.0,78.2%
11.5,79.2%
12.0,80.2%
12.5,81.2%
13.0,82.2%
13.5,83.1%
14.0,83.9%
14.5,84.8%
15.0,85.6%
15.5,86.4%
16.0,87.2%
16.5,87.9%
17.0,88.6%
17.5,89.3%
18.0,89.9%
18.5,90.5%
19.0,91.1%
19.5,91.6%
20.0,92.2%
20.5,92.7%
21.0,93.2%
21.5,93.6%
22.0,94.0%
22.5,94.5%
23.0,94.8%
23.5,95.2%
24.0,95.5%
24.5,95.9%
25.0,96.2%
25.5,96.5%
26.0,96.7%
26.5,97.0%
27.0,97.2%
27.5,97.4%
28.0,97.6%
28.5,97.8%
29.0,98.0%
29.5,98.2%
30.0,98.3%
30.5,98.6%
31.0,98.6%
31.5,98.7%
32.0,98.8%
32.5,98.9%
33.0,99.0%
33.5,99.1%
34.0,99.2%
34.5,99.3%
35.0,99.3%
35.5,99.4%
36.0,99.5%
36.5,99.5%
37.0,99.6%
37.5,99.6%
38.0,99.6%
38.5,99.7%
39.0,99.7%
39.5,99.7%
40.0,99.8%
