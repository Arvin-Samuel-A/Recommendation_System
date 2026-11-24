cd /root/run13
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 venv/bin/python -u /root/early_stop/per_user_eval.py /root/early_stop/per_user_ml $(cat /root/early_stop/ml_arms.txt)
echo DONE exit $?
