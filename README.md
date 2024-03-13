# MetaboliteNER
This repository contains the codes and models of a metabolite named entity recognition (NER) project. The main features you can find in this repository are as follows: 

* An automatically annotated metabolomics training corpus (`TrainingSet.txt`, `TrainingSetAnnot.tsv`) in the [Corpus](Corpus) directory;
* A manually annotated metabolomics test corpus (`GoldStandard.txt`, `GoldStandardAnnot.tsv`) in the [Corpus](Corpus) directory;
* An automatically annotated metabolomics corpus (`MetabolomicsCorpus.txt`, `MetabolomicsCorpusAnnot.tsv`) in the [Corpus](Corpus) directory (which comprises with all TrainingSet data and the automatically annotated results of the test set);
* A rule-based annotation pipeline that automatically annotate ([AutoCORPus](https://github.com/omicsNLP/Auto-CORPus)-processed) publications (`generate_corpus.py`);
* MetaboListem, a machine learning model that recognises metabolite names (`metabolistem_model.py`), adapted from a chemical NER model named [ChemListem](https://bitbucket.org/rscapplications/chemlistem/src/master/);
* TABoLiSTM, a [BERT](https://arxiv.org/abs/1810.04805)-based machine learning model that recognises metabolite names (`tabolistem_model.py`), adapted from MetaboListem.

[![DOI](https://zenodo.org/badge/407162055.svg)](https://zenodo.org/doi/10.5281/zenodo.10581588)
[![DOI:10.1101/2022.02.22.481457](http://img.shields.io/badge/DOI-10.1101/2022.02.22.481457-BE2536.svg)](https://doi.org/10.1101/2022.02.22.481457)
[![DOI:10.3390/metabo12040276](http://img.shields.io/badge/DOI-10.3390/metabo12040276-074F57.svg)](https://doi.org/10.3390/metabo12040276)

## Dependencies
This project is written in Python 3.7.10 and has been tested on Windows 10. Compatibility with other Python versions and systems have not been verified. 

To exploit our package, the following dependencies are required: 
```
transformers == 4.7.0
scikit-learn == 0.24.1
numpy == 1.19.2
pandas == 1.2.3
tensorflow-gpu == 2.4.0
```

In addition, using `generate_corpus` requires an extra package
```
spacy == 3.0.6
```

## MetaboListem
MetaboListem is a machine-learning based metabolite NER algorithm, adapted from a chemical NER model called [ChemListem](https://bitbucket.org/rscapplications/chemlistem/src/master/). 
### Using MetaboListem
This repository includes a trained MetaboListem model in the in the [Models](Models) folder. You can import and load our MetaboListem model as follows:
```
import metabolistem_model
json_path = 'PATH/TO/metabolistem.json'
model_path = 'PATH/TO/metabolistem.h5'

mm = metabolistem_model.MetaboListem()
mm.load(json_path,model_path)
```

Then you can process your text by, e.g., 
```
text='Glucose, glutamine and lactate are the most frequently mentioned metabolites in cancer studies. '
mm.process(text)
```
which would return a list:
```
[(0, 7, 'Glucose'), (9, 18, 'glutamine'), (23, 30, 'lactate')]
```
The items in the output list are tuples with format `(start_idx, end_idx, metabolite)`, where 

* `start_idx` is the position of the start character. 
* `end_idx` is the position of the end character. 
* `metabolite` is the recognised metabolite.

Similarly, you can process a batch of texts by calling 
```
texts=['We found that 1-methyl-6,7-dihydroxy-1,2,3,4-tetrahydroisoquinoline,', 
       'Morphine and hippuric acid were higher in the disease group.']
mm.batchprocess(texts)
```
which results
```
[[(14, 68, '1-methyl-6,7-dihydroxy-1,2,3,4-tetrahydroisoquinoline')], 
 [(0, 8, 'Morphine'), (13, 26, 'hippuric acid')]]
```
that is, a list of lists where each list corresponds to the entity recognition output of each sentence. 

### Training MetaboListem
Before training your own MetaboListem model with our architecture, it is recommended to obtain the 6B 300d pre-trained word embeddings from [GloVe](https://nlp.stanford.edu/projects/glove/); you can get it by downloading `glove.6B.zip` and unzipping `glove.6B.300d.txt`. Albeit being recommended, the word embedding file is optional and you could proceed without it. Other files (words.txt) can be found in the [ChemListem](https://bitbucket.org/rscapplications/chemlistem/src/master/) repository.

The MetaboListem model can be trained like the following example: 
```
import metabolistem_model
mm = metabolistem_model.MetaboListem()
mm.train(textfile,annotfile,glovefile,runname)
```
where the arguments of `mm.train` are:
* `textfile`: the filename of the training text file - e.g. "TrainingSet.txt"
* `annotfile`: the filename of the training annotation file - e.g. "TrainingSetAnnot.tsv"
* `glovefile`: None, or the filename of the GloVe file - e.g. "glove.6B.300d.txt"
* `runname`: Part of the output filenames.


This would produce a trained model, which is constituted of two main files:

* `metabolistem_$RUNNAME.h5`: the keras model
* `metabolistem_$RUNNAME.json`: auxilliary information

Each epoch in the training process also produces a model file on its own; the files are named: 

* `epoch_$EPOCHNUM_$RUNAME.h5`

Only one `json` file would be produced as the auxilliary information is not dependent to epochs. 

## TABoLiSTM
TABoLiSTM is a model adapted from MetaboListem above; in essence, the main difference is that the GloVe word embedding system is replaced with pre-trained BERT embeddings. 
### Using TABoLiSTM

Applying TABoLiSTM models follows a very similar process as MetaboListem. 

The first step is to import and to load the model (available on [zenodo](https://doi.org/10.5281/zenodo.6340001)):
```
import tabolistem_model
json_path='PATH/TO/tabolistem.json'
weight_path='PATH/TO/tabolistem_weights'    ## no suffix

tm = tabolistem_model.TaboListem()
tm.load(json_path,weight_path)
```

Then, like MetaboListem, you can process your text by calling 
```
tm.process(text)
```
which returns a list of tuples `(start_idx, end_idx, entity)`. 

Alternatively, calling 
```
tm.batchprocess(texts)
```
is recommended if multiple strings are to be processed at once for sake of shorter runtime. 

### Training TABoLiSTM

```
import tabolistem_model
tm = tabolistem_model.TaboListem()
tm.train(textfile, annotfile, runname)
```
where the arguments of `tm.train` are
* `textfile`: the filename of the training text file.
* `annotfile`: the filename of the training annotation file.
* `runname`: Part of the output filenames.

## Dataset
The data used in the study consists of Open Access metabolomics publications (n=1,218) from PubMed Central (PMC). Sentences in the corpus are excerpted from Abstract, Method, Result, and Discussion sections of these publications and processed in a rule-based fashion with `generate_corpus.py`.

#### Dataset structure
The metabolomics dataset provided here comprises two files, namely 
* `TrainingSet.txt` containing sentences that mention at least one metabolite
* `TrainingSetAnnot.tsv` containing the positions of the metabolites and the metabolites themselves

Specifically, the two files are structured as the following examples respectively: 
```
PMC2538910  R01008  Citrulline and ornithine, urea cycle intermediates.
```
```
PMC2538910	R01008	0	10	Citrulline
PMC2538910	R01008	15	24	ornithine
PMC2538910	R01008	26	30	urea
```
where `PMC2538910` is the PMC identifier of the source article, and `R01008` is the sentence identifier. The sentence id has three parts: for example in `R01008`, `R` means the Result section, `01` the second subsection and `008` the ninth identified sentence. 

#### Generating dataset
The training set files are generated automatically with the rule-based annotation pipeline `generate_corpus.py`. To use this script, you can type in command lines:
```
python generate_corpus.py -b PATH/TO/PMC -t PATH/TO/SAVEDIR -m PATH/TO/METABOLITEDICT -n OUTPUTNAME -r REGEX -e EXCLUSION
```
which requires 6 arguments: 
* `-b`: Directory of PMC json files processed by [AutoCORPus](https://github.com/omicsNLP/Auto-CORPus)
* `-t`: Output directory
* `-n`: Output filename
* `-m`: Filepath to a file storing a list of metabolite names for exact dictionary matching (e.g. `MetaboliteNames.txt`)
* `-r`: Filepath to a file storing a list of regular expressions for entity recognition (e.g. `RegexList_RuleBasedAnnotation.txt`)
* `-e`: Filepath to a file storing a list of regular expressions to exclude unwanted entities (e.g. `ExclusionList_RuleBasedAnnotation.txt`)

The metabolite names here (`MetaboliteNames.txt`) were downloaded from the [Human Metabolome Database](https://hmdb.ca/downloads) (HMDB). 
