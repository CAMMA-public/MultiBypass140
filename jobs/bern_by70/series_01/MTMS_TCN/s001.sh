sdir="$(pwd)"
cd $MBy140



nvidia-smi

python main_temp.py  -hp hparams/bern_by70/series_01/MTMS_TCN/h001.yaml \
                          -p models \
                          -d labels/bern/labels_by70_splits/ | tee $sdir/s001__SL_BBY70_s01_MTMS_TCN_phase_100_0.log

python main_temp.py  -hp hparams/bern_by70/series_01/MTMS_TCN/h001.yaml \
                          -p models \
                          -d labels/bern/labels_by70_splits/ \
                          -s extract_predictions | tee -a $sdir/s001__SL_BBY70_s01_MTMS_TCN_phase_100_0.log


nvidia-smi