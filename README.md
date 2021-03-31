# digicampus-clustering
Code for representing the student essays in clusters.



### Requirements
Turku-neural-parser-pipeline
Installation guidelines: https://turkunlp.github.io/Turku-neural-parser-pipeline/
Required model: models_fi_tdt_v2.7

### How to run
1. Parse the exam, requires the parser
```
python3 scripts/parse_essays.py --json tp.json 2>/dev/null
```
2. Make a pickle for an exam
```
python3 doc.py --js tp.json --out tp.pickle
```
3. Launch the visualization tool. Choose a port number, default is 6677. To change which pickle files are loaded, change the variable `DIGI_CLUSTERING_DATA` in the script.
```
./run_flask.sh --port NNNN
```