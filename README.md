# brp-ml-model

## Description

This repository contains the code for the ML models used in the Breath Research Project. The code in this repository is based on Python version 3.11 and TensorFlow version 2.15.0. Below is a detailed description of the repository structure and the functions of selected scripts.

## Structure

- **`data/`**  
  Contains datasets collected from sensors. The data is organized into subfolders, separating raw data from labeled data. Raw data can be pre-classified using the available scripts.

- **`graphs/`**  
  Stores graphics generated using the `matplotlib` library.

- **`models/`**  
  Contains source files for the models, including their layer definitions and training functions. To add a new network model, it should inherit from either `AbstractModel` or `SequentialModel`. This ensures compatibility with the training framework used in the repository.

- **`models/saves/`**  
  Stores trained models.

- **`notebooks/`**  
  Contains Jupyter Notebook files for interactive data analysis and visualization.

- **`scripts/`**  
  Includes various utility scripts for data processing, classification, and visualization. Detailed descriptions of selected scripts are provided below:
  - **`categorise_automatically.py`**  
    Automatically tags raw data based on derivative calculations.
  - **`categorise_by_model.py`**  
    Enables preliminary data classification using an already trained model.
  - **`convert_timestamps.py`**  
    Converts time data from original units to consecutive seconds of measurement.
  - **`labeling.py`**  
    A tool for manual data labeling corrections. It allows simultaneous tagging of data from strain gauges and accelerometers. Users can navigate through graphs using left and right arrow keys to shift the time window. Pressing the `M` (multiple) key activates multi-selection mode, and clicking on a sample opens a label selection window.  
  - **`load_data.py`**  
    Provides useful functions for saving and loading data, as well as preparing data for model training. The `prepare_data_for_training()` function in this script allows transforming user-prepared and corrected data into the format required for model training.
  - **`normalization.py`**  
    Implements algorithms for data normalization and moving averages.
  - **`plot.py`**  
    Functions for creating plots.
  - **`transfer_labels.py`**  
    Contains functions for transferring labels from one dataset to another.

- **`main.py`**  
  The main script containing the model training logic. It allows specifying the model to train and the dataset to be used.

## Workflow

1. **Data Preparation**  
   Place your raw data in the appropriate folders within the `data` directory. Then, run the `categorise_automatically.py` or `categorise_by_model.py` script to perform preliminary classification. 

2. **Verification and Correction**  
   The results of this classification can be verified and corrected using the `labeling.py` script. This script provides an interface for manually tagging data and adjusting time intervals as necessary.

3. **Data Formatting**  
   Use the `prepare_data_for_training()` function in `load_data.py` to transform your prepared and corrected data into a format suitable for model training.

4. **Model Training**  
   Select a model from the `models` directory and train it using the prepared dataset. The training process outputs classification accuracy and other metrics upon completion.

## Authors

- **[Damian Jankowski](https://github.com/pingwin02)**
- **[Kacper Karski](https://github.com/JaKarski)**
- **[Filip Krawczak](https://github.com/prosto20025)**
- **[Maciej Szefler](https://github.com/rysiekpol)**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.