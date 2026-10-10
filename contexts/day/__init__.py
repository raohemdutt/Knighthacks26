"""The Day loop (D1): Atlas runs the whole day unasked.

Morning: a plan for today, chosen from what the box already knows (contexts/day/application/planner.py).
Day: the plan against what the cameras and connectors see, with a re-plan when drift breaks it
(contexts/day/application/tracker.py). Evening: the review. All of it as `day.*` events, so the phone's
Now screen, the notifier and tomorrow's plan read the same thing.
"""
