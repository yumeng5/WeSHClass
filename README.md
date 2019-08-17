# WeSHClass

The source code used for Weakly-Supervised Hierarchical Text Classification, published in AAAI 2019.

## Requirements

Before running, you need to first install the required packages by typing following commands:

```
$ pip3 install -r requirements.txt
```

## Quick Start

```
python main.py --dataset ${dataset} --sup_source ${sup_source} --with_eval ${with_eval} --pseudo ${pseudo}
```
where you need to specify the dataset in ```${dataset}```, the weak supervision type in ```${sup_source}``` (could be one of ```['keywords', 'docs']```), the evaluation type in ```${with_eval}``` and the pseudo document generation method in ```${pseudo}```(```bow``` uses bag-of-words method introduced in the CIKM paper. ```lstm``` uses LSTM language model method introduced in the AAAI paper; it generates better-quality pseudo documents, but requires much longer time for training an LSTM language model).

An example run is provided in ```test.sh```, which can be executed by 
```
./test.sh
```

More advanced settings on training and hyperparameters are commented in ```main.py```.

## Inputs

To run the algorithm, you need to provide the following files under the directory ```./${dataset}```:
1. A corpus (```dataset.txt```) that contains all documents to be classified. Each line in ```dataset.txt``` corresponds to one document.
2. Class hierarchy (```label_hier.txt```) that indicates the parent children relationships between classes (each class can have at most one parent class). The first class of each line is the parent class, followed by all its children classes. Tab is used as the delimiter.
3. Weak supervision sources (can be either of the following two) for each leaf class in the class hierarchy:
* Class-related keywords (```keywords.txt```). You need to provide class-related keywords for each leaf class in ```keywords.txt```, where each line begins with the class name (must correspond to that in ```label_hier.txt```), followed by a tab, and then the class-related keywords separated by space. 
* Labeled documents (```doc_id.txt```). You need to provide labeled document ids for each leaf class in ```doc_id.txt```, where each line begins with the class name (must correspond to that in ```label_hier.txt```), followed by a tab, and then document ids in the corpus (starting from ```0```) of the corresponding class separated by space.
4. (Optional) Ground truth labels to be used for evaluation (the provided labels will not be used for training). You need to set the evaluation type argument (```--with_eval ${with_eval}```) correspondingly. 
* If ground truth labels are available for all documents, put all document labels in ```labels.txt``` where the ```i```th line is the class name (must correspond to that in ```label_hier.txt```) for the ```i```th document in ```dataset.txt```.
* If ground truth labels are available for some but not all documents, put the partial labels in ```labels_sub.txt``` where each line begins with the class name (must correspond to that in ```label_hier.txt```), followed by a tab, and then document ids in the corpus (starting from ```0```) of the corresponding class separated by space.
* If no ground truth labels are available, no files are required.

Examples are given under the three dataset directories.

## Outputs

The final results (document labels) will be written in ```./${dataset}/out.txt```, where each line is the class label id for the corresponding document.

Intermediate results (e.g. trained network weights, self-training logs) will be saved under ```./results/${dataset}/${sup_source}/```.

## Running on a New Dataset

To execute the code on a new dataset, you need to 

1. Create a directory named ```${dataset}```.
2. Prepare input files (see the above **Inputs** section).
4. Modify ```main.py``` to accept the new dataset; you need to add ```${dataset}``` to argparse, and then specify parameter settings (e.g. ```update_interval```, ```pretrain_epochs```) for the new dataset.

You can always refer to the example datasets when adapting the code for a new dataset.

## Citations

Please cite the following papers if you find the code helpful for your research.
```
@inproceedings{meng2018weakly,
  title={Weakly-Supervised Neural Text Classification},
  author={Meng, Yu and Shen, Jiaming and Zhang, Chao and Han, Jiawei},
  booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
  pages={983--992},
  year={2018},
  organization={ACM}
}

@inproceedings{meng2019weakly,
  title={Weakly-supervised hierarchical text classification},
  author={Meng, Yu and Shen, Jiaming and Zhang, Chao and Han, Jiawei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={6826--6833},
  year={2019}
}
```
