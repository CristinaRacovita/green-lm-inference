# Green Language Model Inference: Impact of Retrieval-Augmented Generation

This repository contains the materials needed to replicate the experiments described in the study mentioned above. **The methodology was tested only on a Windows machine**.

### Quick start

#### Prerequisites

Before cloning the project and running the experiments, the following programs have to be installed:

- Docker - https://docs.docker.com/desktop/install/windows-install/` needed to run the vector databases
- Ollama - https://ollama.com/download` used to run the language models
- HWiNFO - https://www.hwinfo.com/download/` records the metrics of interest
- Conda - https://www.anaconda.com/download/success` - to manage the Python environment

#### Setup

The setup consists of the following two steps:

1. clone the repository: `git clone https://github.com/CristinaRacovita/green-lm-inference.git`.
2. open a command line terminal in the project's folder and run: `./setup.sh`. This script will create the directories for storing the results, create the Python environment based on [environment.yml](environment.yml) and pull the language models using Ollama.

#### Running the experiments

For ease or reproducibility, we include the option of running all the experiments using a single script. To perform the experiments, follow the steps:

1. open HWiNFO and start collecting measurements each 100ms and store them in a file called **measurements.csv** in the directory results (this is created automatically after following the setup steps)
2. open a terminal in the directory [scripts](./scripts/) and run the command `./experiments_runner.sh`. Be patient because this will take several hours to run, depending on the used machine.
3. stop collecting the measurements, again using the UI of HWiNFO


#### Analyzing the results

1. copy the obtained measurements.csv in each sub-directory under results
2. run each notebook from the [analysis](analysis.sh) directory to obtain the results

**To replicate the methodology as close as possible, we advise running each script called in [experiments_runner.sh](scripts/experiments_runner.sh) individually after restarting the machine and storing the HWiNFO measurements in the associated sub-directory under results.**.

### Project structure

```
├───analysis - contains the notebooks needed to analyze the results
│   └───figures - stores the figures produced during analysis
├───data - contains the used datasets and a directory where the embeddins will be stored
│   ├───datasets
│   │   ├───arguana
│   │   ├───cqadupstack-webmasters
│   │   └───nfcorpus
│   └───embeddings
├───embeddings - code for generating the embeddings
├───retrieval_augmented_generation - code for testing language models and implementing teh RAG
├───scripts - here are located the scripts needed to conduct the experiments and Docker configuration file for Milvus DB
└───vector_databases - code for implementing the index and query operations for each of the three DBs
```
