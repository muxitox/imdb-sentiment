.DEFAULT_GOAL := help

download:
	@rsync -v -r -e ssh \
		--exclude out/ \
		--exclude data/ \
		--exclude __pycache__/ \
		nct01011@dt01.bsc.es:/home/nct01/nct01011/imdb-sentiment/* .

upload:
	@rsync -v -r -e ssh \
		--exclude __pycache__/ \
		--exclude .git/ \
		--exclude notebooks/ \
		--exclude pictures/ \
		--exclude .idea/ \
		--exclude aclImdb/ \
		./ nct01011@dt01.bsc.es:/home/nct01/nct01011/imdb-sentiment

queue-task:
	@ssh nct01011@mt1.bsc.es 'cd /home/nct01/nct01011/imdb-sentiment && ./launchers/launch.sh train'

debug-task:
	@ssh nct01011@mt1.bsc.es 'cd /home/nct01/nct01011/imdb-sentiment && ./launchers/launch.sh debug'

view-queue:
	@ssh -t nct01011@mt1.bsc.es "watch -n1 squeue"

help:
	@echo "run <make [download|upload]> to move files from/to server"
