Implementation of intelligent agents (Simple Reflex, Model-based, Goal-based, and Utility-based) in a Python simulation environment for AI class at UFMA.

The simulation uses the Mesa framework (Python) to model a grid environment where agents must perceive and interact with their surroundings to maintain cleanliness

The project explores four distinct agent types as defined in modern AI literature:
**Simple Reflex Agent**: Decisions are based solely on current percepts (the status of the current cell) using predefined condition-action rules
**Model-based Reflex Agent**: Maintains an internal state to track which parts of the grid have already been explored and cleaned, allowing for more efficient movement
**Goal-based Agent**: Operates with a specific end-state in mind, choosing actions that lead directly to achieving that goal
**Utility-based Agent**: Uses a utility function to evaluate and select actions that provide the highest "satisfaction" or efficiency
