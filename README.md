# Pupil segmentation in Pytorch


## Getting Started

### Structure

- ```model_weight/```: best checkpoint of the training
- ```submission/```: the final subission
- ```code/train.py```: code for model training
- ```code/config.py```: hyperparameter for model training
- ```code/submission.py```: code for generating the final subission


### Training and Testing

To train a network, call ```train.py ```, and fill two arguments training dataset path and output model path just like:

```python3 train.py --training_path ../dataset/  --output_model ../model_weight/UPP.pt``` 

There are two folders ```img ``` and  ```mask ``` adding to training path

Once a model has been trained, you can modify the arch name in ```valid.py ``` line 120 and evaluate it with:

```python3 valid.py```

You can modify the arch name in ```eval.py ``` line 11 and evaluate the score with:

```python3 eval.py```

if you are satisfy the score, call ```submission.py ```, and fill three arguments testing dataset path, submission path, and the inference model just like:

```python3 submission.py --testing_path ../dataset/  --submission_path ../submission/  --model_path ../model_weight/UPP.pt```

There is a folder ```submission ``` adding to testing path

### Training Details

- The model architecture we used is U-Net++ [[paper](https://arxiv.org/pdf/1807.10165.pdf)] [[docs](https://smp.readthedocs.io/en/latest/models.html#id2)] from [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch).
- The encoder backbone we used seresnet101 from [pretrainmodels](https://github.com/Cadene/pretrained-models.pytorch/tree/master), which pretrain from imagenet.
- We use Lovasz loss and binary cross entropy loss from [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) as loss function, and the reciprocal of the pixel ratio of the two classes in the data set is used as the weight. We directly add the losses of the two and average
- Max learing rate = 1e-3, Max epoch = 50, and Weight decay = 1e-4. We use the epoch with the biggest validation IoU as best weight.
- Apply HorizontalFlip, RandomBrightnessContrast, GridDistortion, and Gaussian noise to augment data, and apply torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
- We use Adam as optimizer
