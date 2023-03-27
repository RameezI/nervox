#### Conda 
To upgrade the packages of the current environment based on the content in environment.yml and setup.py

```conda
conda activate dnn-framework
conda env update --file environment.yml --prune
```

#### Datasets
To add a new dataset to dnn-framework

```bash
cd nervox\datasets
tfds new my_dataset  # Create `my_dataset/my_dataset.py` template files 
# Edit the my_dataset.py to implement the your dataset...
cd my_dataset/
