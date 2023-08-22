# tb2csv
Extract Experiment result from Tensorboard event!

** Note that this repository refers https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py **

---
## Requirements
```
pandas == 1.5.3
tensorboard == 2.12.2
```
---

## File Structure
```
├───results
│       SimCLR_4096_whole_27_1e-4_1.csv
│       SimCLR_4096_whole_27_1e-4_SITTING.csv
│
├───result_summaries
│       SimCLR_4096_whole_27_1e-4.csv
│
└───tensorboard
    ├───YYYY-MM-DD_HHMMSS_SimCLR_4096_whole_27_1e-4_1
    │       events.out.tfevents.1692650164.ip-172-31-14-83.19455.0
    │
    └───YYYY-MM-DD_HHMMSS_SimCLR_4096_whole_27_1e-4_SITTING
            events.out.tfevents.1692655588.ip-172-31-14-83.18438.0
```
---
## Command
```
python3 ./tb2csv.py --interest [keyword]
Example) python3 ./tb2csv.py --interest SimCLR_4096_27_1e-4
```
---
