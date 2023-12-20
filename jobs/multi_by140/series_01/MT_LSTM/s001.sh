sdir="$(pwd)"
cd $MBy140



nvidia-smi

python main_temp.py  -hp hparams/multi_by140/series_01/MT_LSTM/h001.yaml \
                          -p models \
                          -d labels/bern/labels_by70_splits/ | tee $sdir/s001__SL_MBY140_s01_MT_LSTM_phase_100_0.log

python main_temp.py  -hp hparams/multi_by140/series_01/MT_LSTM/h001.yaml \
                          -p models \
                          -d labels/bern/labels_by70_splits/ \
                          -s extract_predictions | tee -a $sdir/s001__SL_MBY140_s01_MT_LSTM_phase_100_0.log


nvidia-smi