# roblox-upcoming-forecast

A forecaster for Roblox's Up-and-Coming chart, entered in make.fun's
[roblox-up-and-coming-forecast-01](https://make.fun/bounty/roblox-up-and-coming-forecast-01) bounty.

It learns from the history how far games move on the chart over 1 to 72 hours (where the game at
each place later had been ranked earlier), weights every horizon equally, and turns the last chart
into probabilities per place. The mass the history gives to games not yet on the chart stays unnamed.

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, deterministic. MIT licensed.
