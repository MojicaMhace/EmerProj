# Emergency Project

## How to build .exe file
1. npm install
2. npm run dump
3. npm run serve (optional for quick test)
4. npm run dist (exe file should be in "dist" folder)

## Far-First vs Near-First
- Far-First: Prioritizes rooms that are furthest from exits, assuming they need an earlier start to evacuate safely.
- Near-First: Prioritizes rooms closest to exits to clear high-traffic areas quickly and prevent bottlenecks.

## How does the "Demo Data" simulation work?
- The simulation uses the Artificial Bee Colony (ABC) Algorithm to optimize evacuation routes. It simulates "scout bees" finding the most efficient paths based on the parameters you set in the sidebar, such as colony size and max iterations. The auto generated CSVs are stored on "demo_results"