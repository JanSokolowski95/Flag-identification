# Flag identification

To run this code, you have to have conda installed (I used miniconda).

After installing conda you have to create new conda environment from environment.yml file by running command in your terminal:

``` 
conda env create --name <env_name> --file=environments.yml
```
to create conda environment with `<env_name>` name, which will allow you to run the code.

To activate environment you have to run this command in your terminal:

``` 
conda activate <env_name>
```

Now you will be able to run the code. You can run it in few ways: 

To see help, run:

```python
python .\main.py --help
```

To download the data, run:

```python
python .\main.py download
```

To train the model, run:

```python
python .\main.py train
```

To run the whole pipeline (download data + train the model), run:

```python
python .\main.py pipeline
```

To predict a single country from a flag image, run:

```python
python .\main.py predict --path=/path/to/image.jpg
```

