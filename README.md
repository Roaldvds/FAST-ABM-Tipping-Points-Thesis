# FAST ABM (Flood Adaptation via Social Transformation Agent-based Model)

Author: Thorid Wagenblast

This repository contains the code for FAST ABM. It includes the Python model files, preparatory analysis files and output analysis notebooks. The repository also contains the data underlying the figures.

## Repository Structure

```
├── data                                    # folder containing the data used
    ├── processed   
├── docs                                    # folder containing the file with the model documentaiton       
├── experiments                             # folder containing the python files to run the experiments for the analysis                      
├── notebooks                               # Stata do-file with econometric models
    ├── preparatory_analysis                # Data preparation to generate processed from raw data files
    ├── Results_analysis_experiments.ipynb  # Data analysis and outputs for paper
├── reports                                 # outputs used in the article
    ├── figures                             # figures   
    ├── tables                              # tables
├── results                                 # zipped csv files containing the results of the experiments
├── src                                     # contains the files to run the model
    ├── agent.py                            # contains the agent classes   
    ├── batch_run_model.py                  # setup for paralellised model runs
    ├── datacollection_district.py          # sets up datacollection to collect data within districts   
    ├── datacollection.py                   # sets up datacollection
    ├── functions.py                        # functions   
    ├── model.py                            # contains main model class
    ├── run_model.py                        # setup for non-parallelised model runs   
├── LICENSE.txt                             # License for the code
├── README.me                               # this file
└── setup.py                                # to set up the model

```
## Running the model
To run this complex model, please follow these steps:
1. For simple model runs or testing whether the code works, execture the src/run_model.py. The src/batch_run_model.py does the same but using the mesa batchrunner, paralellising different model runs.
2. To replicate the data used in the publication, the files in the experiments folder need to be run. As relatively large amounts of data are generated, the runs are split up. Since they are still computationally quite demanding, we recommend to run them on a cluster. 

## Citation

If you use this analysis, please cite:
[insert]

## Acknowledgements
This work was supported by the Dutch Research Council NWO VIDI grant number 191015. The survey data collection was supported by the European Research Council (ERC) under the European Union’s Horizon 2020 Research and Innovation Program (grant agreement number: 758014). 

## Contact
t.wagenblast@tudelft.nl, t.filatova@tudelft.nl
