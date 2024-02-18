These are some json results from endpoints for documentation purposes.

blacklist-02-10-2024: https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/calendar/blacklist, accessed 02-10-2024 (it changes over time)

blacklist-02-17-2024: https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/calendar/blacklist, accessed 02-17-2024 (it changes over time)

season-02-10-2024: https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2023?lang=en&region=us Accessed 02-10-2024

- note that the type shows postseason which matches the season type at time of access.

season-02-17-2024: https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2023?lang=en&region=us Accssed 02-10-2024

- note that the top-level `type` shows regular season but its not the regular season at time of access (its offseason). I assumed (but my code doesnt) that this signified the current season type but in this case it doesnt.

event-event-401547378: https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/401547378?lang=en&region=us. 2023 super bowl (in 2024). Noticed its SF VS KC with KC technically being home (but not SF @ KC)

event-event-401547464: https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/401547464?lang=en&region=us. 2023 regular season game. Noticed shortname is CAR @ DET

teams-8: https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2023/teams/8?lang=en&region=us. detroit lions
