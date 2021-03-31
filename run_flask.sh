export FLASK_ENV=development
export FLASK_APP=flask_app
export DIGI_CLUSTERING_DATA=$(pwd)
export DIGI_CLUSTERING_ROOT=""

# default is port 6677 but you can
# add a new --port NNNN when calling run_flask.sh and that will be used
flask run --port 6677 $*
