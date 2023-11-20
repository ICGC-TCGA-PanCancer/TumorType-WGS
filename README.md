# TumorType-WGS
Classifying tumor types based on Whole Genome Sequencing (WGS) data. A web version of the model where vcf files can be uploaded and predict cancer types can be found at: [DeepTumour](deeptumour.oicr.on.ca)

#### Training RF Models
```bash
$ Rscript train_models.R <dataType> <cancerType>
```
dataType is one of: SNV, SV, CNV, MUT, GEN, PTW, IND
cancerType is one of the 24 cancer types.

#### Training DNN Models
```bash
$ python train_models_tumour_classifier.py <fold> <path/to/features>
```
