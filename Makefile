.DEFAULT_GOAL := help

download:
	@rsync -v -r -e ssh \
		--exclude out/ \
		--exclude data/ \
		--exclude __pycache__/ \
		nct01013@dt01.bsc.es:/home/nct01/nct01013/imdb-sentiment/* .

upload:
	@rsync -v -r -e ssh \
		--exclude __pycache__/ \
		--exclude .git/ \
		--exclude notebooks/ \
		--exclude pictures/ \
		--exclude .idea/ \
		--exclude aclImdb/ \
		./ nct01013@dt01.bsc.es:/home/nct01/nct01013/imdb-sentiment

queue-task:
	@ssh nct01013@mt1.bsc.es 'cd /home/nct01/nct01013/imdb-sentiment && ./launchers/launch.sh train'

debug-task:
	@ssh nct01013@mt1.bsc.es 'cd /home/nct01/nct01013/imdb-sentiment && ./launchers/launch.sh debug'

view-queue:
	@ssh -t nct01013@mt1.bsc.es "watch -n1 squeue"

help:
	@echo "run <make [download|upload]> to move files from/to server"
