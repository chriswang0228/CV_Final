# Pupil segmentation in Pytorch


## Getting Started

Structure:
- ```dataset/img```: all images in S1~S4
- ```dataset/mask```: all masks in S1~S4
- ```model_weight/```: best checkpoint of the training
- ```evaluation/```: all the model mIoU in test set
- ```l_curve/```: model learning curve
- ```submission/```: the final subission
- ```code/train.py```: code for model training
- ```code/config.py```: hyperparameter for model training
- ```code/valid.py```: code for calculating the mIoU
- ```code/eval.py```: code for evaluate the score
- ```code/submission.py```: code for generating the final subission

#### Dataset
Download the full S1 to S8 dataset in ```dataset/img```, and copy the S1 to S4 images and masks to ```dataset/img``` and ```dataset/mask```

#### Training and Testing

To train a network, modify the ```config.py ``` and call:

```python train.py ``` 

Once a model has been trained, you can modify the arch name in ```valid.py ``` line 120 and evaluate it with:

```python valid.py```

You can modify the arch name in ```eval.py ``` line 11 and evaluate the score with:

```python eval.py```

if you are satisfy the score, you can modify the arch name in ```submission.py ``` line 11 and the exp name in ```submission.py ``` line 12 and generate the final submission with:

```python submission.py```
