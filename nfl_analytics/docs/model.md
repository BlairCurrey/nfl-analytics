# Model

The model itself is a simple linear regression model. This was intended as a starting point with many different avenues for improvement, but it's performance isnt too bad. The mean absolute error is around 10. This is the average amount the predicted spread is off by. I could not find an average difference for Vegas odds but according to [Vegas Always Knows? A Mathematical Deep Dive](https://www.theonlycolors.com/2020/9/29/21492301/vegas-always-knows-a-mathematical-deep-dive) the standard deviation of Vegas spreads is around 14. That is, 68% of games are within 14 points of the spread, and 32% are higher. Thus, a mean absolute error of 10 seems like we're in the ballpark even if not really competitive.

## Ideas for Improvement

1. Accounting for relationships between features such as home pass offense compared to away pass defense. This could be approached from a few directions:

- Use a different model architecture that can pick up on these or train a deep neural network.
- Keep using linear regression and engineer features such as net home passing yard average (and similar), which would be the difference between the home team's average passing yards and the away team's average passing yards allowed. This way we can capture the relationship despite all the features being treated as independent, as linear regression models do.

2. Account for injuries. Currently this model just probably shouldn't be used on games where there are meaningful injuries. Injuries could be quantified as missing win shares, which is [a measure that seeks to measure how much a player contributes to the outcome of a game](https://www.nfl.com/news/2023-nfl-season-projecting-win-share-leaders-on-offense-quarterbacks-non-qbs-and). This brings several complexities which makes you appreciate how sophisticated the professional models for this must be. These complexities include:

- getting historical data. This could probably be done using the same nfl playbook pdfs that nflfastR get's its play by play data from.
- getting player injury statuses for upcoming games. I am unsure of a reliable datasource for this. Each additional datasource adds some failure point which should also be handled gracefully.
- Determining how likely a player is to play based off their injury status (questionable, out, likely, etc.) and other factors. This is a complex problem in itself. Alternatively, the model could be trained with these statuses and their associated missing winshares if the data is available.

3. Do not predict the spread at all. Instead, train a classifier to detect games that are likely to be outliers. That is, predict when the spread will be outside the standard deviation of (supposedly) 14 points, and in what direction (overdog under or overperforming). While I have doubts about the possible accuracy you could achieve with such a model, I suspect there may be more opportunity to find outliers than to simply beat Vegas at predicting the spread.
