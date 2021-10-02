# KusuLLC

## Installation

This project uses [DVC](https://dvc.org/) for model and data version control.

1. `$git clone https://github.com/sundrelingam/KusuLLC.git`.

    a) From the top level directory of the repository, create a Python virtual enivornment with (optional):

		$cd \path\to\KusuLLC
		$python -m venv \path\to\venv
		
	b) Install required packages with `$pip install -r requirements.txt`

2. Install DVC [for your operating system](https://dvc.org/doc/install).
3. Add remote storage with `$dvc remote add -d storage gdrive://xyz`. Please contact @sundrelingam for access to remote storage.
4. `$dvc pull`.

## Getting Started

Follow these steps to run the modules. The instructions are given for Windows:

1. Activate the virtual environment with `$\path\to\venv\Scripts\activate`
2. Run models such as:

```
$python -c "from sentiment_analysis import Sentiment; Sentiment().analyze('PINS')"
```

This repository has 3 models for evaluating your stock picks.

1. **Sentiment** / `Sentiment().analyze("PINS")`:

    In order to run this model, you will need credentials for the Reddit API. Simple instructions can be found here: https://towardsdatascience.com/scraping-reddit-data-1c0af3040768.
    
    You will also need twitter developer access. Simple instructions can be found here: https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/twitter-data-in-python/

2. **Fundamentals** / `Fundamentals().analyze("PINS")`:
    
    This module relies on data from `SimFin` which is downloaded to the `./models/data/simfin` directory. If you do not want to re-pull this data and write it to disk, use `Fundamentals().analyze("PINS", update_data=False)`.

3. **Technicals** / `Technicals().analyze("PINS")`:
    
    Uses a deep neural network trained on a variety of technical indicators calculated on historical stock prices to predict price direction.

## TL;DR

All the above models can be run using the `HYDRA` wrapper as follows:

```
Hydra("PINS")
```

## Additional Resources

For additional "Kusu" based investing advice, please read [The Kusu Investor](https://docs.google.com/document/d/1rCxaUnPtl0IX3S19bTvKNj58g32ywV0KJtv--sGWaGk/edit?usp=drivesdk) currently in its first edition (by invite only).
