"""Module containing Observer classes.

Observers are a way to track and keep hierarchy over games and agents. They
are also tasked withebatching move/value training data.

Contains:
- BaseObserver: Headless observer with the minimum for controlling self play.
- UIObserver: TKinter based visualization enabling human interaction.
"""
