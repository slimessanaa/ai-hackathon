# Hex AI Hackathon

Minimal Hex agent for the Game AI Hackathon.

Reached the semifinals.

## Main files

- `agent_v7.py` - latest agent version
- `agent.py` - submission-style agent entrypoint
- `arena.py`, `bestof3_arena.py`, `varied_arena.py` - local test helpers
- `README_AGENT.md` - quick local usage notes
- `README_STRATEGY_TESTING.md` - strategy and testing notes

## Idea

The agent combines fast Hex-specific heuristics:

- immediate win and block checks
- shortest-path evaluation
- bridge-aware virtual connections
- defensive pressure against the opponent path
- shallow reply checking for the best candidate moves

## Result

This project made it to the hackathon semifinals.
