# Green Language Model Inference: Impact of Retrieval-Augmented Generation

This repository contains the materials needed to replicate the experiments described in the study mentioned above. **The methodology was tested only on a machine with Windows 10**.

### Quick start

#### Prerequisites

Before cloning the project and running the experiments, the following programs have to be installed:

- Docker - https://docs.docker.com/desktop/install/windows-install/ needed to run the vector databases
- Ollama - https://ollama.com/download used to run the language models
- HWiNFO - https://www.hwinfo.com/download/ records the metrics of interest
- Conda - https://www.anaconda.com/download/success to manage the Python environment

#### Setup

It consists of the following two steps:

1. Clone the repository with `git clone https://github.com/CristinaRacovita/green-lm-inference.git`.
2. Open a command line terminal in the project's folder and run: `./setup.sh`. This script will create the directories for storing the results, install the Python environment based on [environment.yml](environment.yml) and pull the language models using Ollama.

### Running the experiments

For ease of reproducibility, we include the option of running all the experiments using a single script. To perform the experiments, follow the steps:

1. Open HWiNFO and start collecting measurements every 100ms and store them in a file called **measurements.csv** in the directory results (this is created automatically after following the setup steps).
2. Open a terminal in the directory [scripts](./scripts/) and run the command `./experiments_runner.sh`. Be patient because this will take several hours to run, depending on the used machine.
3. Stop collecting the measurements, again using the UI of HWiNFO.

### Analyzing the results

1. Copy the obtained measurements.csv in each sub-directory under results.
2. Run each notebook from the [analysis](./analysis/) directory to obtain the results.

**To replicate the methodology as closely as possible, we advise running each script called in [experiments_runner.sh](scripts/experiments_runner.sh) individually after restarting the machine. Do not forget to follow the methodology presented above for storing the associated HWiNFO measurements, mentioning that now the measurement.csv has to be stored in the corresponding sub-directory for each experiment under results.**.

### Project structure

The project directories are described below:

```
├───analysis - contains the notebooks needed to analyze the results
│   └───figures - stores the plots produced during analysis
├───data - contains the used datasets and a directory where the embeddings will be stored
│   ├───datasets
│   │   ├───arguana
│   │   ├───cqadupstack-webmasters
│   │   └───nfcorpus
│   └───embeddings
├───embeddings - code for generating the embeddings
├───retrieval_augmented_generation - code for testing language models and implementing the RAG
├───scripts - scripts needed to conduct experiments and Docker configuration file for Milvus DB
└───vector_databases - code for implementing the index and query operations for each DB
```
