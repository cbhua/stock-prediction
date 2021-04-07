# :chart_with_upwards_trend: Stock Prediction Model

A model about stock price prediction by using machine learning.

## :bulb: Introduction

One hundred percent precise stock prediction is ... everyone's dream, we can use machine learning to do ... anyway, let's directly introduce to our model. 

This project is based on LSTM structure to predict the one-line time series stock trend. 

## :wrench: Envirenment 

``` Python
Python Version: 3.6 or later
Python Packages: torch, numpy, matplotlib, sklearn
```

## :file_folder: Structure

### Folders

- `data`: datasets will be here
    - `ori`: the original dataset files, examples here are `.csv` files
    - `trans`: the transferred dataset files, would be `.npy` files, the `train.py` process will do it 
- `save`: trained model will be saved here
- `src`: all source code will be here
    - `models`: all models' class file will be here
    - `utils`: tool functions like dataset transfer will be here 
- `fig`: all results figs and images for readme will be here

### Files

- `train.py`: the process for trinning the model, trained model will be saved 
- `test.py`: the process for test the model, the model will be loaded from `save` folder
- `config.ini`: all hyperpremeters and process controls will be here, just modify a single file 
- `README.md`: you are reading now

## :floppy_disk: How to use

Step 1. Download or Clone this repository. 

Step 2. Run `train.py` to transfer the example original dataset to numpy files and then use it to train the model. After trainning, the model will be saved in `save` folder. 

Step 3. Run `test.py` to test the example, the running result will be saved as figs in `fig` folder. 

Step 4. After testing the example, you can modify the `congif.ini` to test different parameters and generate your results. And modify modules to get what you want. 

## :bar_chart: Examples

