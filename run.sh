python -u main.py --dataset=Mafengwo --predictor=MLP --loss_type=BPR --learning_rate=0.0001 --device=cuda:3 --num_negatives=8 --layers=3 --epoch=200

python -u main.py --dataset=CAMRa2011 --predictor=DOT --loss_type=BPR --learning_rate=0.001 --device=cuda:2 --num_negatives=2 --layers=2 --epoch=30
