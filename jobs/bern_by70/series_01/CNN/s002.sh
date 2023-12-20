sdir="$(pwd)"
cd $MBy140



nvidia-smi

python main_cnn.py  -hp hparams/bern_by70/series_01/CNN/h002.yaml \
                          -p models \
                          -d labels/bern/labels_by70_splits/ | tee $sdir/s002__SL_BBY70_s01_CNN_step_100_0.log

python main_cnn.py  -hp hparams/bern_by70/series_01/CNN/h002.yaml \
                          -p models \
                          -d labels/bern/labels_by70_splits/ \
                          -s extract_predictions | tee -a $sdir/s002__SL_BBY70_s01_CNN_step_100_0.log


nvidia-smi