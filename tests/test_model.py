import torch
import unittest
from torch.utils.data import DataLoader

from models.extractor.hcnn import HCNN
from models.extractor.vcnn import VCNN
from models.predictors.ANNPredictor import ANNPredictor
from models.predictors.LSTMPredictor import LSTMPredictor
from models import Predictor, GlacierModel
from datasets import Glacier_dmdt, ERA5Datasets, GlacierDataset
from runner import trainer


class PredictorModelTest(unittest.TestCase):
    def test_lstmModel(self):
        tensor = torch.randn(32, 256)
        model = LSTMPredictor(input_dim=256, hidden_dim=256, n_layers=2, bidirection=True, p=0.5)
        out = model(tensor)
        self.assertEqual(out.shape, (32, 1))

    def test_lstmModel1(self):
        tensor = torch.randn(32, 256)
        model = LSTMPredictor(input_dim=256, hidden_dim=256, n_layers=1, bidirection=True, p=0.5)
        out = model(tensor)
        self.assertEqual(out.shape, (32, 1))

    def test_annModel(self):
        tensor = torch.randn(32, 256)
        model = ANNPredictor(input_dim=256, hidden_dim=256, n_layers=2, bidirection=True, p=0.5)
        out = model(tensor)
        self.assertEqual(out.shape, (32, 1))

    def test_predictor(self):
        tensor = torch.randn(32, 256)
        model = Predictor(input_dim=256, hidden_dim=256, n_layers=2, bidirection=True, p=0.5, layers=[
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        ])
        out = model(tensor)
        self.assertEqual(out.shape, (32, 1))


class DataLoaderTest(unittest.TestCase):
    def test_DMDTLoader(self):
        smb = Glacier_dmdt("JAKOBSHAVN_ISBRAE", 1980, 2002, path="../glaicer_dmdt.csv")
        self.assertEqual(len(smb), 2002 - 1980)

    def test_ERA5Datasets(self):
        ERA5 = ERA5Datasets("JAKOBSHAVN_ISBRAE", 1980, 2002, path="../ECMWF_reanalysis_data")
        self.assertEqual(len(ERA5), 2002 - 1980)
        data = ERA5[0]
        self.assertEqual(data.shape, (5, 289, 12))

    def test_DataLoader(self):
        smb = Glacier_dmdt("JAKOBSHAVN_ISBRAE", 1980, 2002, path="../glaicer_dmdt.csv")
        data = ERA5Datasets("JAKOBSHAVN_ISBRAE", 1980, 2002, path="../ECMWF_reanalysis_data")
        dataset = GlacierDataset([data, data], [smb, smb])
        loader = DataLoader(dataset, batch_size=1)
        for feature, s in loader:
            self.assertEqual(feature.shape, (1, 5, 289, 12))
            self.assertEqual(s.shape, (1,))


class FeatureExtractorModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        smb = Glacier_dmdt("JAKOBSHAVN_ISBRAE", 1980, 2002, path="../glaicer_dmdt.csv")
        data = ERA5Datasets("JAKOBSHAVN_ISBRAE", 1980, 2002, path="../ECMWF_reanalysis_data")
        dataset = GlacierDataset([data], [smb])
        cls.loader = DataLoader(dataset, batch_size=16)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.train_loader = cls.loader
        cls.test_loader = cls.loader

    def test_hcnn(self):
        model = HCNN(output_dim=256, vertical_dim=289)
        for feature, s in self.loader:
            out = model(feature)
            self.assertEqual(out.shape, (16, 256))
            break

    def test_vcnn(self):
        model = VCNN(output_dim=256)
        for feature, s in self.loader:
            out = model(feature)
            self.assertEqual(out.shape, (16, 256))
            break

    def test_hcnn_lstm1(self):
        extra = VCNN(output_dim=256)
        pred = LSTMPredictor(input_dim=256, hidden_dim=256, n_layers=1, bidirection=True, p=0.5)
        model = GlacierModel(extra=extra, pred=pred)
        for feature, s in self.loader:
            out = model(feature)
            self.assertEqual((16, 1), out.shape)
            break

    def test_hcnn_lstm2(self):
        extra = VCNN(output_dim=256)
        pred = LSTMPredictor(input_dim=256, hidden_dim=256, n_layers=2, bidirection=True, p=0.5)
        model = GlacierModel(extra=extra, pred=pred)
        for feature, s in self.loader:
            out = model(feature)
            self.assertEqual((16, 1), out.shape)
            break

    def test_hcnn_lstm3(self):
        extra = VCNN(output_dim=256)
        pred = LSTMPredictor(input_dim=256, hidden_dim=256, n_layers=3, bidirection=True, p=0.5)
        model = GlacierModel(extra=extra, pred=pred)
        for feature, s in self.loader:
            out = model(feature)
            self.assertEqual((16, 1), out.shape)
            break

    def test_runner_vcnnlstm(self):
        vcnn_model = VCNN(in_channel=5, output_dim=256, vertical_dim=289)
        lstm_model = LSTMPredictor(layers=None, input_dim=256, hidden_dim=256, n_layers=1, bidirection=False, p=0.5)
        trainer(vcnn_model, lstm_model, train_loader=self.train_loader, test_loader=self.test_loader,
                device=self.device, epochs=1, lr=0.002, reg=0.001, save_every=10, print_every=10,
                loss_func=torch.nn.MSELoss,
                optimizer=torch.optim.Adam,
                save_path="saved_models/VCNN_LSTM_model.h5")

    def test_runner_hcnnlstm(self):
        hcnn_model = HCNN(in_channel=5, output_dim=256, vertical_dim=289)
        lstm_model = LSTMPredictor(layers=None, input_dim=256, hidden_dim=256, n_layers=1, bidirection=False, p=0.5)
        trainer(hcnn_model, lstm_model, train_loader=self.train_loader, test_loader=self.test_loader,
                device=self.device, epochs=1, lr=0.002, reg=0.001, save_every=10, print_every=10,
                loss_func=torch.nn.MSELoss,
                optimizer=torch.optim.Adam,
                save_path="saved_models/HCNN_LSTM_model.h5")

    def test_runner_vcnnann(self):
        vcnn_model = VCNN(in_channel=5, output_dim=256, vertical_dim=289)
        ann_model = ANNPredictor(layers=None, input_dim=256, hidden_dim=256, n_layers=1, bidirection=False, p=0.5)
        trainer(vcnn_model, ann_model, train_loader=self.train_loader, test_loader=self.test_loader,
                device=self.device, epochs=1, lr=0.002, reg=0.001, save_every=10, print_every=10,
                loss_func=torch.nn.MSELoss,
                optimizer=torch.optim.Adam,
                save_path="saved_models/VCNN_ANN_model.h5")

    def test_runner_hcnnann(self):
        hcnn_model = HCNN(in_channel=5, output_dim=256, vertical_dim=289)
        ann_model = ANNPredictor(layers=None, input_dim=256, hidden_dim=256, n_layers=1, bidirection=False, p=0.5)
        trainer(hcnn_model, ann_model, train_loader=self.train_loader, test_loader=self.test_loader,
                device=self.device, epochs=1, lr=0.002, reg=0.001, save_every=10, print_every=10,
                loss_func=torch.nn.MSELoss,
                optimizer=torch.optim.Adam,
                save_path="saved_models/HCNN_ANN_model.h5")


if __name__ == '__main__':
    unittest.main(verbosity=0)
