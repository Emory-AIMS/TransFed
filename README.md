# 🏥 TransFed: Semi-Supervised Federated Transfer Learning for Healthcare

### Official repository for TransFed (submitted to ECML-PKDD 2025).

📄 Paper coming soon.

Healthcare institutions often need to collaborate on developing predictive models while adhering to privacy regulations and handling heterogeneous data collection practices. Traditional federated learning approaches assume shared feature spaces or patient populations across institutions, limiting their applicability in real-world healthcare settings where different institutions collect distinct sets of patient data. We propose TransFed, a novel semi-supervised federated transfer learning framework that enables effective collaboration across healthcare institutions with heterogeneous feature spaces. Our framework combines cross-domain feature alignment with semi-supervised learning to leverage both labeled and unlabeled data, while maintaining privacy through federated learning principles. Using two large real-world clinical datasets, we demonstrate that TransFed effectively enables knowledge transfer  without requiring direct data sharing or common feature spaces to improve prediction performance across domains and generalizes well to unseen healthcare systems.

## Problem setting

<img width="621" alt="problem setting" src="https://github.com/user-attachments/assets/6905c20d-8c5f-4c97-9b9c-4725e260dc3d" />

TransFed is applicable in the federated transfer learning scenario illustrated above, where the source and target domains share overlapping features but have disjoint sample sets. The source domain is fully labeled, while the target domain contains only partial labels. The learned models can also be applied to previously unseen domains.

## Training pipeline

The sample training procedure is demonstrated below, the detailed explanations can be found in the paper.

<img width="621" alt="procedure" src="https://github.com/user-attachments/assets/211540d4-28fc-4db6-9777-e1ba23b6b624" />

## Requirement

Install the environment:

<pre>conda env create -f environment.yml
conda activate transfed
</pre>

## Running TransFed

TransFed can be ran using the following argument:

Run setupdata.py to prepare dataloader for the target domain first, then run

<pre>bash python transfed.py \ 
    --lambda_sc 0.0001 \ 
    --lambda_u 5</pre>

where lambda_sc is the weight for supervised contrastive loss, lambda_u is the weight for pseudo-labeling loss. We found that the weight for consistency loss is robust at 0.0001 for our experimental datasets so it's not included for arguments.

For a full training run example with recommended hyperparameters, see [`train.sh`](train.sh).

## Datasets

- eICU Collaborative Research Database

- PhysioNet Sepsis Prediction Challenge 2019

For eICU Collaborative Research Database, we conducted some data pre-processing to create the 69 features (as detailed in the code and the paper). First, we take the last measurement on labs.csv, which records labs measurements, for each lab each "patientunitstayid", and then drop features with over 30% missing values. Then, we join the processed labs dataset on "patientunitstayid" with other datasets: patients.csv which takes patients' personal information, vitalperiodic.csv and vitalaperiodic.csv that take vital signs, apacheapsvar.csv that takes APACHE severity of illness measures, hospital.csv that records hospital information. Then, categorical features are on-hot encoded, and missing values are filled with the mean of each feature. After pre-processing, the final dataset contains 69 features, and the target feature is `unitdischargestatus', which is the discharge status of the patient from the ICU unit stay.

For PhysioNet Sepsis Prediction Challenge 2019, we did similar pre-processing: for each feature, we take the last measurement and fill the missing value with the mean. 

## Results

<img width="497" alt="Screenshot 2025-05-10 at 4 10 57 PM" src="https://github.com/user-attachments/assets/0143e38e-eb17-4c2d-a9e1-a0a5db86d8be" />

The table summarizes the models evaluated on data from unseen domains, $C_\text{M}$ (Midwest) and $C_\text{W}$ (West).
Model $f_A$, trained on Domain A (South)'s local data, is tested on Domain C's data using features corresponding to those of Domain A. Similarly, Model $f_B$, trained on Domain B (Northeast)'s local data, is tested on Domain C's data using features corresponding to those of Domain B.

More results and detailed discussions are available in the [paper (coming soon)]().

## Contributors

Linghui Zeng, Ruixuan Liu, Li Xiong, Joyce C. Ho

Emory University

## Acknowledgments

The research is supported by National Science Foundation under grants 2124104, 2125530, 2302968, 2145411 and National Institute of Health under R01ES033241 and R01LM013712.
