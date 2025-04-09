import numpy as np
import pandas as pd

# Set up data for Domain B limited labeled data

# Load datasets
datab_labeled = pd.read_csv("datab_labeled.csv")

shared_feats = [
    'heartrate', 'respiration', 'noninvasivesystolic', 'noninvasivediastolic',
    'noninvasivemean', 'admissionheight', 'dischargeweight','motor',
    'verbal', 'gender_Unknown', 'ethnicity_Asian', 'gender_Other', 
    'ethnicity_Native American', 'ethnicity_African American', 'ethnicity_Hispanic', 
    'gender_Male', 'gender_Female','unitdischargestatus'
]

acols = [
  'Hct', 'chloride', 'Hgb', 'RBC', 'admissionheight', 'dischargeweight',
  'calcium', 'platelets x 1000', 'MCV', 'bicarbonate', 'RDW', 'AST (SGOT)',
  'ALT (SGPT)', 'total protein', 'alkaline phos.', 'magnesium', '-basos',
  'total bilirubin', '-polys', 'respiration', 'noninvasivesystolic',
  'noninvasivediastolic', 'noninvasivemean', 'intubated', 'vent', 'dialysis',
  'verbal', 'meds', 'urine', 'wbc', 'respiratoryrate', 'motor',
  'ph', 'hematocrit', 'bun', 'bilirubin', 'creatinine', 'heartrate',
  'albumin', 'sodium', 'ethnicity_African American',
  'ethnicity_Asian', 'ethnicity_Caucasian', 'ethnicity_Hispanic',
  'ethnicity_Native American', 'ethnicity_Other/Unknown', 'gender_Female',
  'gender_Male', 'gender_Other', 'gender_Unknown'
]

bcols = [
  'BUN', 'potassium', 'WBC x 1000', 'heartrate','respiration','noninvasivesystolic',
  'glucose', 'meanbp', 'admissionheight', 'noninvasivediastolic', 'noninvasivemean',
  'dischargeweight', 'anion gap', 'MCH', 'MCHC', '-lymphs', '-monos', '-eos',
  'fio2', 'observationoffset', 'age','sao2', 'eyes', 'motor', 'pao2', 'pco2',
  'ethnicity_African American','verbal',
  'ethnicity_Asian', 'ethnicity_Caucasian', 'ethnicity_Hispanic',
  'ethnicity_Native American', 'ethnicity_Other/Unknown', 'gender_Female',
  'gender_Male', 'gender_Other', 'gender_Unknown'
]


# Extract anchors and labels directly from labeled dataset
anchors_labeled = datab_labeled[bcols].to_numpy()
anchors_labels_labeled = datab_labeled["unitdischargestatus"].to_numpy()

np.savez_compressed(
    '5fold/fold1/anchors_labeled.npz',
    anchors=anchors_labeled,
    anchors_labels=anchors_labels_labeled
)

print("saved anchors_labeled.npz")

