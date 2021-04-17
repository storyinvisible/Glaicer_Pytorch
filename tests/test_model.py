import torch
import unittest
from torch.utils.data import DataLoader

from models.extractor.hcnn import HCNN
from models.extractor.vcnn import VCNN
from models.predictors.ANNPredictor import ANNPredictor
from models.predictors.LSTMPredictor import LSTMPredictor
from models.predictors.predictor import Predictor
from datasets import Glacier_dmdt, ERA5Datasets, GlacierDataset


class PredictorModelTest(unittest.TestCase):
    def test_lstmModel(self):
        tensor = torch.randn(32, 256)
        model = LSTMPredictor(input_dim=256, hidden_dim=256, n_layers=2, bi_direction=True, p=0.5)
        out = model(tensor)
        self.assertEqual(out.shape, (32, 1))

    def test_annModel(self):
        tensor = torch.randn(32, 256)
        model = ANNPredictor(input_dim=256, hidden_dim=256, n_layers=2, bi_direction=True, p=0.5)
        out = model(tensor)
        self.assertEqual(out.shape, (32, 1))

    def test_predictor(self):
        tensor = torch.randn(32, 256)
        model = Predictor(input_dim=256, hidden_dim=256, n_layers=2, bi_direction=True, p=0.5, layers=[
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
        for i in range(5):
            self.assertEqual(data[i].shape, (1, 289, 12))

    def test_DataLoader(self):
        smb1 = Glacier_dmdt("JAKOBSHAVN_ISBRAE", 1980, 2002, path="../glaicer_dmdt.csv")
        data1 = ERA5Datasets("JAKOBSHAVN_ISBRAE", 1980, 2002, path="../ECMWF_reanalysis_data")
        smb2 = Glacier_dmdt("JAKOBSHAVN_ISBRAE", 1980, 2002, path="../glaicer_dmdt.csv")
        data2 = ERA5Datasets("JAKOBSHAVN_ISBRAE", 1980, 2002, path="../ECMWF_reanalysis_data")
        dataset = GlacierDataset([data1, data2], [smb1, smb2])
        loader = DataLoader(dataset, batch_size=1)
        for feature, s in loader:
            self.assertEqual(feature[0].shape, (1, 1, 289, 12))
            self.assertEqual(feature[1].shape, (1, 1, 289, 12))
            self.assertEqual(feature[2].shape, (1, 1, 289, 12))
            self.assertEqual(feature[3].shape, (1, 1, 289, 12))
            self.assertEqual(feature[4].shape, (1, 1, 289, 12))
            self.assertEqual(s.shape, (1,))


class FeatureExtractorModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        smb = Glacier_dmdt("JAKOBSHAVN_ISBRAE", 1980, 2002, path="../glaicer_dmdt.csv")
        data = ERA5Datasets("JAKOBSHAVN_ISBRAE", 1980, 2002, path="../ECMWF_reanalysis_data")
        dataset = GlacierDataset([data], [smb])
        cls.loader = DataLoader(dataset, batch_size=16)

    def test_hcnn(self):
        model = HCNN(output_dim=256)
        for feature, s in self.loader:
            tensor = torch.cat(feature, dim=1)
            out = model(tensor)
            self.assertEqual(out.shape, (16, 256))
            break

    def test_vcnn(self):
        model = VCNN(output_dim=256)
        for feature, s in self.loader:
            tensor = torch.cat(feature, dim=1)
            out = model(tensor)
            self.assertEqual(out.shape, (16, 256))
            break


if __name__ == '__main__':
    unittest.main(verbosity=0)
