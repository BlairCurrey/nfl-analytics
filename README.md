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
- getting nflfastR data into python: https://gist.github.com/Deryck97/dff8d33e9f841568201a2a0d5519ac5e
- nfl game books: https://support.nfl.com/hc/en-us/articles/4989073286044-Game-Books
  - official nfl game stats published as pdf. should be able to find on the nfl game summary page such as https://www.nfl.com/games/texans-at-ravens-2023-post-2?active-tab=watch

# TODO:

- [x] setup venv, install dependencies, handle dependencies (poetry? requirements.txt?)
  - using poetry. love it compared to alternatives so far.
- [x] script to get data from github repo
- [x] script to load into pd data frame
- [x] identify required stats for predicting spread. start simple.
  - rush/pass yards offense/defense, points scored/allowed, actual spread
- [ ] setup workflow for jupyter notebook
  - [x] install ipykernel and use venv/kernal in notebook. can use this get venv location `poetry show -v`
  - [ ] refactor data get/load into functions i can import.
    - path error... relative paths not resolving when importing
- [ ] aggregate game stats from play by play for each game, for each season.
  - model: Season table (or dataframe) with each row being a game. WeekNumber, Team1, Team2, Team1Score, Team2Score, Team1PassYards, Team2PassYards, etc.
    - could be Home and Away team instead of 1,2
    - not sure if this should be dataframe or sqlite table maybe pandas but also sqlite for dev purposes (if still not using notebook).
- [ ] use aggregated team stats to get average stats for each team coming into each game. this will be
  - should be able to use to query something like `"SELECT AVERAGE(pass_yards) as avg_pass_yards, etc. FROM Season where Team1 = 'CHI' or Team2 = 'CHI' and week_number < 10"`
    - this would be the input to the model to predict the week 10 game for the chicago bears
    - this is going to be a lot of duplicate aggregation if I do this for each week. may be a good way to use previously computed averages. maybe storing totals and counts for each stat in a separate table and then computing averages from that table.

# Current status:

- project managed by poetry
- Can fetch all play by play data in compressed csv format from ` nflfastR`` releases using  `get_data.py``
- Can load raw data into pandas dataframe and sqlite3 using load_data.py. data types are inferred by pandas and sqlite is created from pandas method. this is key because there are a lot of columns (372 at time of writing). When identifying actual columns I want to use I can get more specific with data types.
