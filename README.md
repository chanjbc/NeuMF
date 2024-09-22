# NeuMF Movie Recommendation System

This repository contains a PyTorch implementation of the Neural Matrix Factorization (NeuMF) model, introduced by [Xiangnan He et al.](http://dx.doi.org/10.1145/3038912.3052569), for movie recommendations using the MovieLens [32M](https://grouplens.org/datasets/movielens/32m/) and [100K "latest-small"](https://grouplens.org/datasets/movielens/latest/) datasets. The project includes scripts for data preparation, model training, evaluation, and inference.

## Authors
- [James Chan](https://github.com/chanjbc)


## Table of Contents

1. [Requirements](#requirements)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Downloading the Dataset](#downloading-the-dataset)
   - [Training the Model](#training-the-model)
   - [Running Inference](#running-inference)
5. [File Descriptions](#file-descriptions)
6. [Contributing](#contributing)
7. [License](#license)

## Requirements

- Python 3.7+
- PyTorch
- pandas
- numpy
- tqdm

## Project Structure

```
.
├── data/
|   ├── ml-latest-small
|   |   └── ...
|   └── ml-32m
|       └── ...
├── encodings/
|   ├── encodings/encoding-ml-latest-small-sample.pkl
|   └── encodings/encoding-ml-32m-sample.pkl
├── models/
|   ├── model-ml-latest-small-sample.pth
|   └── model-ml-32m-sample.pth
├── Download.py
├── PrepareData.py
├── NeuMF.py
├── TrainEvaluate.py
├── main.py
├── Inference.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/neumf-movie-recommendation.git
   cd neumf-movie-recommendation
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### OPTIONAL: Downloading the Dataset

This step is optional since running `main.py` will automatically check for downloaded datasets and fetch one if not already present.

To download and extract either the MovieLens 32M or 100K dataset, run:

```bash
python Download.py <dataset>
```

Replace `<dataset>` with either `ml-32m` or `ml-latest-small`.

Again, note that this repository only has support for the MovieLens [32M](https://grouplens.org/datasets/movielens/32m/) and [100K "latest-small"](https://grouplens.org/datasets/movielens/latest/) datasets.

### Training the Model

To train the NeuMF model on the chosen dataset, run:

```bash
python main.py <dataset>
```

Replace `<dataset>` with either `ml-32m` or `ml-latest-small`.

This will prepare the data, train the model, and save both the model and encoders with timestamped filenames in the `models/` and `encodings/` directories, respectively.

The ml-latest-small 100K dataset trains significantly faster than the full ml-32m dataset, making it preferred for getting started or for prototyping your recommendation system. You can always switch to the full dataset later for more comprehensive results.

### Running Inference

To generate movie recommendations for a new user, first add movie titles to `user_movies` in `Inference.py`, ensuring that their names match the titles listed in the corresponding dataset's `movies.csv`. The existing example is:

```python
user_movies = [
   "How to Train Your Dragon (2010)",
   "How to Train Your Dragon 2 (2014)",
   "Lord of the Rings: The Fellowship of the Ring, The (2001)",
   "Sound of Music, The (1965)",
   "That Darn Cat (1997)",
   "Enchanted (2007)"
]
```

Then use `Inference.py`:

```bash
python Inference.py <dataset> <top_n> <encoding_file> <movies_file>
```

Replace the placeholders with the appropriate file paths:
- `<dataset>`: dataset used (i.e., `ml-32m` or `ml-latest-small`)
- `<top_n>`: number of recommendations to return (e.g., 20)
- `<encoding_path>`: encoder file (e.g., `encoding-ml-32m-2024-09-21-12-00-00.pkl`)
- `<model_path>`: trained model file (e.g., `model-ml-32m-2024-09-21-12-00-00.pth`)

Two encoding files, `encoding-ml-latest-small-sample.pkl` and `encoding-ml-32m-sample.pkl`, as well as their corresponding model files `model-ml-latest-small-sample.pth` and `model-ml-32m-sample.pth`, are available for testing. To use these files, first ensure that the corresponding dataset has been downloaded with `Download.py`, and then run either of these commands:

- ml-latest-small:
   ```bash
   python Inference.py ml-latest-small 20 encoding-ml-latest-small-sample.pkl model-ml-latest-small-sample.pth
   ```

- ml-32m:
   ```bash
   python Inference.py ml-32m 20 encoding-ml-32m-sample.pkl model-ml-32m-sample.pth
   ```

## File Descriptions

- `Download.py`: Script to download and extract the MovieLens dataset.
- `PrepareData.py`: Contains functions for data preparation and loading.
- `NeuMF.py`: Defines the Neural Matrix Factorization model architecture.
- `TrainEvaluate.py`: Includes functions for training and evaluating the model.
- `main.py`: The main script for running the entire training process.
- `Inference.py`: Script for generating movie recommendations for new users.

## Contributing

Contributions to this project are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
