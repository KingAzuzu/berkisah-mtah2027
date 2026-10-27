Title: [Team <TEAM_NAME>] Three rule questions before we build

1. Label convention. We train on the public IDSSE/DFL matches (approved in #16). DFL's ball-status and
possession channels may define the start and end of a dead ball, and contested-ball ownership,
differently from your corrected tracking. May we read the public 5 minutes to understand your
convention (for example, whether D starts when the ball crosses the line or at the whistle) and
encode that understanding as a rule, provided no number is fitted to those 5 minutes?

2. Camera framing. The tactical camera moves, and the per-frame homography tells us where it points.
May we use the camera's pointing direction, derived by an algorithm from the provided homography,
as a feature for where the ball is? We want to be sure the operator's framing does not count as
"human-produced data about the game".

3. Committing model outputs. We may run a public ball-action spotting model (T-DEED, trained on
SoccerNet, i.e. other games) on the provided video using a GPU. GPU inference is not bit-exact, so,
following #14, we would commit the model's predictions (timestamp, action class, side) together with
the script and the checkpoint version. Does committing these predictions to a public repository
count as publishing derived data? They contain no positions and no video.
