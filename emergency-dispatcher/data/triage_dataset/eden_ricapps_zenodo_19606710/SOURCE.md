# Dataset source and preparation notes

- Title: Clinical-functional classification of emergency department presenting complaints based on the Manchester Triage System (EDEN-RICAPPS)
- Authors: Susana García-Gutierrez, Maria Jose Legarreta Olabarrieta, Irantzu Barrio Beraza
- Repository: [Zenodo record 19606710](https://zenodo.org/records/19606710)
- DOI: [10.5281/zenodo.19606710](https://doi.org/10.5281/zenodo.19606710)
- Version: 1.0, published 2026-04-16
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), confirmed by the Zenodo API record and the included README/data dictionary/classification rules.
- Provenance: Derived methodological output from the EDEN-RICAPPS project. Dataset documentation states it contains no patient-level data or outcomes.
- Intended scope: complaint/diagnostic code to one of eight broad clinical-functional chapters. The documentation explicitly says this is not an official Manchester algorithm, does not grade severity within a chapter, and is not for bedside clinical decision-making.

## Downloaded files

- `classification_manchester_ed_presenting_complaints.xlsx` (169,398 bytes; MD5 `06d98681e081be0110816d453f1900b7`)
- `README.MD` (3,835 bytes; MD5 `6e7987dea082e0e286574978f036a59d`)
- `Data Dictionary.pdf` (285,440 bytes; MD5 `4a8138b205940311ddc5bd491c9b57bf`)
- `Classification Rules.pdf` (347,420 bytes; MD5 `d1c570b97884d3d85ed7084d0655e47a`)

The workbook has one sheet, 6,989 data rows, and columns `identificador`, `MotivoConsultaMATSET_EDIT`, `cat_manchester_def`, and `Categoria_Manchester`; no cells are missing in these columns. The identifier column is excluded. After case normalization there are two codes with conflicting chapter assignments (`V54.9` and `V58.9`); their four source mapping rows are recorded in `excluded_ambiguous_codes.csv` and excluded from training. Repeated mappings are reduced to one row per unambiguous normalized code in `complaint_code_chapter.csv` (1,089 unique codes). Class counts in this prepared file are in `../../triage_models/eden_ricapps_code_chapter_evaluation.json`.

The [Mendeley candidate `dd359z6trk`](https://data.mendeley.com/datasets/dd359z6trk/2) advertised CC BY 4.0 and expert triage labels, but its public archive URL returned S3 `AccessDenied`; no files were saved from that source. [PhysioNet MIETIC](https://www.physionet.org/content/mietic/1.0.0/) requires credentialing, CITI training, and a signed data-use agreement, so restricted files were not accessed. The Zenodo data above were selected only for the narrower code-to-chapter research demonstration, not as free-text symptom/severity labels.

The cleaned CSV is derived from the downloaded workbook using:

```powershell
.\.venv312\Scripts\python.exe scripts\prepare_eden_ricapps.py
```

Use `--force` only when intentionally regenerating the derived files.



