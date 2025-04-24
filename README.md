# RBMIRM
RBMIRM is designed using residual bottleneck blocks to estimate 11 leaf nutrient parameters from multispectral drone images. This is the repository for my research article titled "Precise Estimation of Rice Leaf Macro and Micro Nutrients using a Residual Block-based Multi-spectral Image Regression Model". This repository contains the PyTorch implementation of our proposed RBMIRM model for nutrient estimation from multi-spectral drone images. The code supports training and testing on crop leaf data, generating nutrient prediction results and visualizing MAE and error metrics through boxplots.

# Requirements
This project was developed and tested with the following setup:
- Python 3.10.12
- PyTorch 1.13.0 with CUDA 11.7
- OS: Linux (Ubuntu)

# Dataset
- **LeafSampleAnalysis.csv** – Contains ground truth values for training (11 nutrients [B, Ca, Cu, Fe, K, Mg, Mn, Na, P, S, Zn]).
- **LeafSampleAnalysis_t.csv** – Contains ground truth values for testing.
- **Image.txt** – List of training image filenames.
- **Image_t.txt** – List of testing image filenames.  
  Ensure that the paths to all input files (CSV and TXT) are correctly set in `loaddata_withindex.py` before running.
- **Image/** folder – Contains multispectral image samples (5 bands).  
  Make sure to adjust the image folder path in `loaddata_withindex.py` according to your local directory.

# Running the Code
1. **Train and Test the Model**  
   ```bash
   python3 date_wise.py
This script will:
- Train the RBMIRM model using training data
- Test it on unseen data
- Save output and target values to 4 Excel files for both training and testing
  
2. **Generate Datewise (Growthstage wise) Loss/Error Files**
   ```bash
   python3 datewise_boxplot.py
Generates training and testing loss/error files.

3. **Plot MAE Boxplots**
    ```bash
   python3 d.py
Displays boxplots and mean absolute errors (MAE) for all 11 predicted nutrients.


