"""The League (Hemdutt): the friends' weekly game on top of Habits, Day Run, Waste Watch and the Circle.

    cfg = load_config(Path("config/league.yaml"))
    league = League(cfg, circle_id="team", me="hemdutt", names={...}, sched=habits.sched)
    league.subscribe(bus)                      # pure projection: rebuilt from the log (replay=True)
    bus.subscribe("play.league.week.closed", league.on_week_closed_live)   # live only: proposes the cards

Every week (local Monday to Sunday) each member's daily summaries are scored against the targets in
config/league.yaml. The first event after the boundary, in event time, closes the week with a
`play.league.week.closed` event, and the box then *proposes* a GroupMe post and a Circle plan for
the stake. Nothing is sent without an approval. Only daily summaries travel; never raw signals.
"""

from contexts.play.league.application.league import League
from contexts.play.league.config import LeagueConfig, load_config

__all__ = ["League", "LeagueConfig", "load_config"]
