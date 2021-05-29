# KusuLLC

## Installation

This project uses [DVC](https://dvc.org/) for model and data version control.

1. `$git clone https://github.com/sundrelingam/KusuLLC.git`.

    a) Create a Python virtual enivornment with (optional):

		```
		python -m venv /some/path
		source /some/path/bin/activate
		```
		
	b) Install required packages with `$pip install -r requirements.txt`

2. Install DVC [for your operating system](https://dvc.org/doc/install).
3. Add remote storage with `$dvc remote add -d storage gdrive://xyz`. Please contact @sundrelingam for access to remote storage.
4. `$dvc pull`.

## Getting Started

This repository has 2 models for evaluating your stock picks.

1. `Sentiment`:

In order to run this model, you will need credentials for the Reddit API. Simple instructions can be found here: https://towardsdatascience.com/scraping-reddit-data-1c0af3040768.

The module can be run as follows: `Sentiment().analyze("PINS")`.

2. `Fundamentals`:

This module can be run as follows: `Fundamentals().analyze("PINS")`.

This module relies on data from `SimFin` which is downloaded to the `./models/data/simfin` directory. If you do not want to re-pull this data and write it to disk, use `Fundamentals().analyze("PINS" update_data=False)`.