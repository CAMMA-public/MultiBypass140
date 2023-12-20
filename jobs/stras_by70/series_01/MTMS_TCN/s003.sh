sdir="$(pwd)"
cd $MBy140



nvidia-smi

python main_temp.py  -hp hparams/stras_by70/series_01/MTMS_TCN/h003.yaml \
                          -p models \
                          -d labels/strasbourg/labels_by70_splits/ | tee $sdir/s003__SL_SBY70_s01_MTMS_TCN_phase_step_100_0.log

python main_temp.py  -hp hparams/stras_by70/series_01/MTMS_TCN/h003.yaml \
                          -p models \
                          -d labels/strasbourg/labels_by70_splits/ \
                          -s extract_predictions | tee -a $sdir/s003__SL_SBY70_s01_MTMS_TCN_phase_step_100_0.log


nvidia-smi