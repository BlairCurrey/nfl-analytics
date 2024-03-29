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
- a concise little overview on features from a datascience.exchange comment about predicting matches (NOTE: not spread): https://datascience.stackexchange.com/questions/102827/how-to-predict-the-winner-of-a-future-sports-match
- article on when you need to scale data for ml: https://www.baeldung.com/cs/normalization-vs-standardization
- some espn api documentation: https://gist.github.com/nntrn/ee26cb2a0716de0947a0a4e9a157bc1c

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
- [x] aggregate game stats from play by play for each game, for each season. keep the features simple for now.
  - [x] not sure if this should be dataframe or sqlite table maybe pandas but also sqlite for dev purposes (if still not using notebook).
    - dataframe for sure to make it easier to develop.
  - [x] figure out how to model it
    - Games dataframe with:
      - Team, pass off, pass def, rush off, rush def, average EPA, score spread
- [x] use aggregated team stats to get average stats for each team coming into each game. this will be

score differential is wrong? look at first game. the number for the 2 teams dont match

- [x] cleanup
  - [x] the get_data and load_data is duplicated in data.py and get_data.py/load_data.py. just use one or the other.
  - [x] move notebook code to python files. think about a managable way to share logic between notebook and python files so I can drop into the pipeline and inspect as needed.
    - probably just put everything in a functions that are imported into the python file and notebook?
- [x] simple model to predict spread
  - [x] use sklearn to train model
    - could do linear regression, although maybe not ideal because relationship between features (pass off vs. opponent pass def)
    - gradient boosting (better with non-linear)?
  - [x] some sort of basic analysis to see how it performed. including manually comparing to vegas spread (maybe I can find an average difference? https://www.theonlycolors.com/2020/9/29/21492301/vegas-always-knows-a-mathematical-deep-dive)
    - 9-10 pt avg difference (?). normaly distrubution means ~68% will be within 1 std deviation (identified as 14-15). could be a little lower because 1,2, etc. are within 14-15, but could be higher because ~32% will be more than 14-15.
- [x] add function to create matchups from 2 teams so we can predict next week's games.
  - using the running_avg df to merge, similar to how we're merging the game_id to get the final training df
  - in practice the merged records should share a week but in theory they could be different (week 12 detroit vs. week 6 ravens etc.).
- [x] cli
  - [x] download data
  - [x] train model
    - what to do with it? save configuration then recreate it when needed? pickle?
  - [x] predict spread
- [x] github action pipeline basic
  - [x] setup manual run that downloads/trains/uploads artifacts
    - [x] add save functionality to --train flag that saves the running_avg_df to assets
    - [x] update predict fn to only predict from this saved df. should ensure its always using latest data that model was trained with (instead of using new data model wasnt trained with when building from csv).
- Github actions improvements
  - [ ] remove draft status when ready?
  - [x] add simple static body with explanation. what this is/contains, how to consume, etc.
  - [ ] rethink release setup. currently it publishes a new set with timestamp to one spread-predictor release. its going to accumulate a lot of stuff and the timestamps are not very easy to understand.
    - override the same release? meh dont love that
    - new release for each? spread-predicter-[timestamp] (w/ or w/out timestamp on files)
    - use year/week instead of timestamp?
    - or, just overide old releases? do I ever need the old ones?
  - [x] add error exits to cli util. job should fail if --download, --train etc. fails
  - [ ] run on some schedule. end of day tuesday between season start and season end?
  - [ ] output model mean absolute error to md file and use as body of release (with any other info). have some release template.
  - [0] (maybe) if there are any hardcoded paths (like asset dir?), think about how to not hardcode them.
    - punting on this one. not really important to make this configurable.
- Quality of Life Improvements
  - [ ] rename model? LinRegSpreadPredictor? in the release at least, not sure if anywhere else (pipeline)
    - LinReg is descriptive but is it an implementation detail. Do I want to have an DecisionTreeSpreadPredictor in the future? Or would I only have a decision tree based model if it replaced the lin reg one? Maybe thats a "wait until (if) you actually have another model" problem.
  - [ ] suppress pandas warnings?? "import pandas as pd"
  - [ ] add cli doc generator. look into `argparse.HelpFormatter` to generate a markdown file.
  - [x] add types
  - [ ] unit tests
- [ ] Model improvements
  - [ ] W/L record or games played and win pct? (win and loss column on game aggregation)
  - [ ] success rate (calculate success (0 or 1) from each play).
    - could be measured from positive EPA. or 40% of yards to go on 1st, 70% on 2nd, 100% on 3rd/4th (or similar)
    - actually it looks like there are some success rate fields?
  - [x] total points scored/allowed
  - [ ] maybe dont use first ~3 games? small sample size but dont want to throw out too much data.
  - [ ] games played (could be used as confidence in record/stats)
- [x] rethink exposing build_running_avg_dataframe, build_training_dataframe instead of doing that inside train_model (with side effect of saving the build_running_avg_dataframe (to disk?) somewhere).
  - just need to see how its actually used
  - I guess its good for development purposes? maybe just make the df arg in train_model(df) optional and build from ground up if not provided which will be used in cli/deployment but developing can pass it df? idk
  - keeping as it makes it a bit easier to develop in notebook (maybe) and its just not that important to put in the train fn. although the idea that it will always be done for training seems correct.
- [x] write script that gets upcoming games and makes prediction from model
  - try to find a good source for the schedule (nflfastR for that too maybe?).
  - [x] matchups from schedule (like `[{home: 'DET', away: 'CHI'}, ...]`)
  - [x] expose this on cli (with no predict, it gets upcoming games?)
  - [x] add step to ci that predicts upcoming games
- [ ] upcoming prediction website
  - [ ] use script and some template to generate html page with predictions.
  - [ ] publish to github pages (use seperate branch to commit dist? look how i did it for https://github.com/BlairCurrey/peggle-clone)

# Next steps (beyond TODO):

- "Blogpost" (or github md readme.)
- Possible change of direction: multilabel classifer to predict if a matchup is normal or an outlier if overdog wins by a lot more than expected or underdog performs much better than expected. Then use that to pick the over/underdog to cover the spread (or to pass on the game)
  - Not sure of labels (overdog_performance: normal, underperform, overperform).
  - Find games where spread (included in dataset) is off and then train a model to classify them. Input can be the same team stats as the spread predictor, but the dataset will be limited to just the games that are off.
  - outlier definition tbd. perhaps abs(real spread - booky spread) is > 1 std deviation? which is 14-15 points according to some article I found (in resource section). In that case I'd expect ~68% of games to be "normal" and ~32% to be outliers.
  - Crazy idea: After implmeneting, go back and use this as a feature on the spread. Classify each games as (normal,outperform,underperform) and use that as a feature to train with. Perhaps this will be useful even if the accuracy is somewhat low?
    - probably a bad idea. i think this will essentially leak training data into test data

# Stray thoughts:

- model name idea: caliper. (like measuring the "spread")
- save model by pickeling with joblib/dump or save the configuration
- I think saving the configuration is probably better if I can.

- What should the model's be guess _exactly_ and what does that say about how the teams are modeled in the input? the spread consists of 2 numbers (usually the inverse of each). 1 for each team. Maybe just predict the hometeam?
  - probably need to squash 2 teams into 1 line like: home_team_pass_off, home_team_pass_def, away_team_pass_off, away_team_pass_def, etc.
- Are lots of features bad? What about redundant or mostly redundant features (pass yards, rush yards, total yards (total yards are either equal or very similar to pass+rush yards)). Which should I pick in that case (probably the less aggregated ones)?

# Modeling ideas

## Running Averages By Week

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

Then I could do something like SELECT pass_off_running_avg FROM Season where Team = 'CHI' week = 3 to get the averages for a given week.

## Final input

Maybe something like this where:

- spread is for home team
- everything on 1 line with home and away team stats (running average from season start)
- maybe even exclude home/away team? shouldnt really factor them in right? would home field advantage be a factor here?
  GameID,Date,HomeTeam,AwayTeam,HomePassOff,AwayPassOff,HomeRushOff,AwayRushOff,HomePassDef,AwayPassDef,HomeRushDef,AwayRushDef,Spread
  1,2022-09-10,TeamA,TeamB,250,220,120,100,180,200,90,110,3
  2,2022-09-10,TeamC,TeamD,280,240,130,110,200,180,95,100,-7
  3,2022-09-11,TeamE,TeamF,220,200,110,90,170,190,85,120,5

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

# some potential schedule sources

from: https://gist.github.com/nntrn/ee26cb2a0716de0947a0a4e9a157bc1c

Main goal is to get a list of all team pairs (identified as home/away) for an upcoming game at a given point in time.

## start/end date for each week for current season

https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/calendar/blacklist

- not exactly sure what makes it a blacklist (maybe their internal purposes). the "blacklist" appears to contains days (although not every day and probably not just game days) within the season
- this "blacklist" contains details on the weeks for preseason, regular season, post season.
- no team matchups, just defines the week start/end
- for the current season only

## current season details (preseason? regular? post? off?)

https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2023?lang=en&region=us

- probably a good place to check current status of the league ("are we in season and if so is it pre/regular/post)
- top level "type" seems to focus on the current item in "types". For example, `types.items` is an array of objects like:

```json
[
  { "type": 1, "name": "Preseason", "startDate": "...", "endDate": "..." },
  { "type": 2, "name": "Regular Season" },
  { "type": 3, "name": "Postseason" },
  { "type": 4, "name": "Off Season" }
]
```

and `type` is the particular number where the actual date is within the start/end date. In other words, if its the off season `type` will be the offseason item.

## weeks by year

https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2023/types/2/weeks?lang=en&region=us

- gives refs for the different weeks in a given year

## get "events" by week/sreason/season type

https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2023/types/2/weeks/18/events?lang=en&region=us

- just refs. not teams mentioned
- 1: pre, 2: reg, 3: post, 4: off

## event details

can be gotten from the `v2/sports/football/leagues/nfl/seasons/2023/types/2/weeks/18/events` endpoint
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/401547647?lang=en&region=us
can identify teams with one of:

```json
  "name": "Buffalo Bills at Miami Dolphins",
  "shortName": "BUF @ MIA",
```

# error on predict

When predicting teams that did not have results from the previous week. For example, when running this aftert he superbowl:

    poetry run python nfl_analytics/main.py --predict det ch

I see this error:

File "/Users/blaircurrey/Library/Caches/pypoetry/virtualenvs/nfl-analytics-i6gxw6-C-py3.12/lib/python3.12/site-packages/sklearn/utils/validation.py", line 1035, in check_array
raise ValueError(
ValueError: Found array with 0 feature(s) (shape=(1, 0)) while a minimum of 1 is required by StandardScaler.

This is "normal" in the sense that its a bad input and we dont (shouldnt) make a prediction. But its not handled well. Should return a better sort of error.
