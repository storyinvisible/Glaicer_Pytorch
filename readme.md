# Glacier Model
Global ice melting has become one of the most significant problems nowadays. It will give rise to a lot of issues, such as sea-level rise, climate anomaly, etc. Both Arctic and Antarctic ice sheets are losing mass at accelerating rates. Greenland is one of the major ice storage districts. It has the second largest ice coverage in the world. In recent years, ice melting is growing abnormally in Greenland, and there is also a huge climate change in the area of ice melting in Greenland. 

We aimed to develop a machine learning model that can estimate the change in Mass Balance of the glaciers based on the regional climate that has an impact on the surface ice melt and ocean climate that impact ice calving. As each glacier is uniquely different from each other in term of themselves and the how the environment impact on their total mass loss since these environmental-specific factors cannot be capture by the current mathematical model, we hypothesize that machine learning model can capture these differences.

## Environment setup

The model requires pytorch 1.8.1. It is tested on CUDA 10.2 environment.

To setup the environment you can visit [Start Locally | PyTorch](https://pytorch.org/get-started/locally/)

For windows device simply run:

```bash
pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

For linux/macOS environment:

```bash
pip3 install torch torchvision torchaudio
```

Then for other required packets, simply run:

```bash
pip3 install -r requirements.txt
```
## To simply test the model using previously saved models
```
python app.py -model saved_models_2D\HCNNLSTM-STORST\HCNNLSTM-STORST_model.h5 -glacier STORSTROMMEN -year 2000
```

## File structure

```
.
├── datasets.py
├── models
│   ├── __init__.py
│   │   ├── __init__.py
│   │   ├── hinverted.py
│   │   ├── hcnn.py
│   │   ├── tcnn.py
│   │   ├── twcnn.py
│   │   ├── vcnn.py
│   │   └── vinverted.py
│   ├── extractor
│   └── predictors
│       ├── __init__.py
│       ├── ANNPredictor.py
│       └── LSTMPredictor.py
├── plots
├── README.md
├── requirements.txt
├── runner.py
├── saved_models
└── utils.py
```

## Models

Our models are divided into two different parts, an extractor combined with an predictor.

You can combine a feature extractor with a predictor to construct a full glacier model using:

```python
glacier_model = GlacierModel(extractor, predictor, name="myGlacierModel")
```

### Extractors

The extractor is supposed to extract the features from the feature dataset and give out extracted embedding features.

To use the model, simply follow the example below:

```python
model = VCNN(in_channel=5, output_dim=256, vertical_dim=289)
```

To add your own model, you follow the same structure as:

```python
class ExtractorModel(nn.Module):
    """ 
    Parameters:
    - in_channel: input data channels
    - output_dim: latent size of linear layer
    """
    def __init__(self, in_channel=5, output_dim=256, **args):
        super(ExtractorModel, self).__init__()
        self.args = args
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(17340, output_dim)
        )
    def forward(self, x):
        return self.model(x)
```

### Predictors

The predictor is taken the extracted feature from the extractors and make prediction based on the feature vector.

To use an predictor, you can follow:

```python
model = LSTMPredictor(layers=None, input_dim=256, hidden_dim=256, n_layers=1, bidirection=False, p=0.5)
```

To add your own predictor model, you can follow the same structure as:

```python
class ANNPredictor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, p=0.5, **args):
        super(ANNPredictor, self).__init__()
        self.args = args
        self.predictor = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=hidden_dim, out_features=1),
        )
    def forward(self, x):
        return self.predictor(x)
```

## Model training:

To train the model, you can simply make use of the trainer function from `runner.py`.

The model takes a full combined glacier model with a extractor and a predictor.

The train_loader will give pairs of feature vectors and target values for the model to regression.

The `testdataset` and `testsmb` is used as the feature vector for the test data and test target.

To can set the learning rate using the `lr` parameter, and the regularization term `reg` takes an float number to penalize the model parameters.

You can define a loss function first, and pass the loss function to the trainer function. For the optimizer, simply pass the optimizer class without initialized will be fine.

If you want to same the model for future prediction, you can pass the directory to same the model folder using: `save_path`

```python
trainer(model, train_loader=loader, testdataset=testdata, testsmb=testsmb, show=False, device=cuda, epochs=3, lr=0.002, reg=0.001, save_every=10, eval_every=1, test_split_at=15, critic=loss_function, optimizer=torch.optim.Adam, save_path="saved_models")
```