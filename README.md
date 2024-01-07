# factoriai
Factorio is a factory building game centred around automation, but after finishing the game, what is there left ot automate? The answer: THE GAME ITSELF!

Factorai my pet project of automating the game using AI. It is split into three components: the data capture script, the model training code and the heuristic AI. 

The heuristic AI is intended to generate additional (although suboptimal) gameplay to serve as a baseline for the AI.

# usage

Currently this repo has only implemented the data capture and heuristic bot (albeit this has not been pushed yet). To use the data capture script, run the following commands from the root directory:

`
pip installs -r requirements.txt
python scripts/capture.py
`
After 15 seconds, the script will begin taking screenshots and recording keystrokes and mouse positions. The capture data will be saved inside the data folder. To turn off the capture, press f9.

All sessions are assigned uniquely hashed names from the POSIX time and machine MAC. Thus, data contributors can merge data by simply pushing the generated changes to github.
