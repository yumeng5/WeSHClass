# WeSHClass

The source code used for Weakly-Supervised Hierarchical Text Classification, published in AAAI 2019.

## Requirments

Before running, you need to first install the required packages by typing following commands:

```
$ pip3 install -r requirements.txt
```

## Running

```
python main.py --dataset ${dataset} --sup_source ${sup_source}
```
where you need to specify the dataset in ```${dataset}```, the weak supervision type in ```${sup_source}``` (could be one of ```['keywords', 'docs']```).

An example run is provided in ```test.sh```, which can be executed by 
```
./test.sh
```

More advanced settings on training and hyperparameters are commented in ```main.py```.

## Inputs

The weak supervision sources ```${sup_source}``` can come from any of the following:
* Class-related keywords (```keywords```); you need to provide class-related keywords for each class in ```./${dataset}/keywords.txt```, where each line begins with the class id (starting from ```0```), followed by a colon, and then the class-related keywords separated by commas. 
* Labeled documents (```docs```); you need to provide labeled document ids for each class in ```./${dataset}/doc_id.txt```, where each line begins with the class id (starting from ```0```), followed by a colon, and then document ids in the corpus (starting from ```0```) of the corresponding class separated by commas. 

Examples are given under the three dataset directories.

## Outputs

The final results (document labels) will be written in ```./${dataset}/out.txt```, where each line is the class label id for the corresponding document.

Intermediate results (e.g. trained network weights, self-training logs) will be saved under ```./results/${dataset}/${model}/```.

## Running on a New Dataset

To execute the code on a new dataset, you need to 

1. Create a directory named ```${dataset}```.
2. Put raw corpus (with or without true labels) under ```./${dataset}```.
3. Modify the function ```read_file``` in ```load_data.py``` so that it returns a list of documents in variable ```data```, and corresponding true labels in variable ```y``` (If ground truth labels are not available, simply return ```y = None```).
4. Modify ```main.py``` to accept the new dataset; you need to add ```${dataset}``` to argparse, and then specify parameter settings (e.g. ```update_interval```, ```pretrain_epochs```) for the new dataset.

You can always refer to the example datasets when adapting the code for a new dataset.

## Citations
