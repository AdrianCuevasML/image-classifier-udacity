#test
import unittest
from unittest.mock import patch, MagicMock
#module
from image_classifier.train_functions import TrainFunctions
#local
import os
import tempfile
import sys
from io import StringIO
#third party
import torch
from torchvision import models
from torch.utils.data import DataLoader


class TestTrainFunctions(unittest.TestCase):

    def setUp(self):
        self.data_dir = "fake_dir"
        self.train_functions = TrainFunctions(data_dir=self.data_dir)

    @patch('image_classifier.train_functions.Loader')
    @patch('torchvision.datasets.ImageFolder')
    def test_load_data(self, mock_image_folder, mock_loader):
        # Configura los mocks
        mock_train_data = MagicMock()
        mock_valid_data = MagicMock()
        mock_test_data = MagicMock()
        mock_image_folder.side_effect = [mock_train_data, mock_valid_data, mock_test_data]

        mock_train_loader = MagicMock()
        mock_valid_loader = MagicMock()
        mock_test_loader = MagicMock()
        mock_loader.side_effect = [mock_train_loader, mock_valid_loader, mock_test_loader]

        train_data, train_loader, valid_loader, test_loader = self.train_functions.load_data()

        mock_image_folder.assert_any_call('fake_train_dir', transform=self.train_functions._transform_data()[0])
        mock_image_folder.assert_any_call('fake_valid_dir', transform=self.train_functions._transform_data()[1])
        mock_image_folder.assert_any_call('fake_test_dir', transform=self.train_functions._transform_data()[2])


        mock_loader.assert_any_call(mock_train_data, batch_size=32, shuffle=True)
        mock_loader.assert_any_call(mock_valid_data, batch_size=32)
        mock_loader.assert_any_call(mock_test_data, batch_size=32)

        self.assertIs(train_data, mock_train_data)
        self.assertIs(train_loader, mock_train_loader)
        self.assertIs(valid_loader, mock_valid_loader)
        self.assertIs(test_loader, mock_test_loader)

    def test_create_model(self):
        """Test the create_model function."""
        vgg16_model = self.train_functions.create_model(arch="vgg16", hidden_units=512)
        resnet18_model = self.train_functions.create_model(arch="resnet18", hidden_units=512)

        self.assertIsInstance(vgg16_model, models.VGG)
        self.assertIsInstance(resnet18_model, models.ResNet)

    def test_train_model(self):
        """Test the train_model function."""
        model = models.vgg16(pretrained=True)
        train_loader = DataLoader(torch.utils.data.TensorDataset(torch.randn(100, 3, 224, 224), torch.randint(0, 10, (100,))))

        self.train_functions.train_model(model, train_loader, train_loader, learning_rate=0.001, epochs=1, gpu=False)

        original_stdout = sys.stdout
        sys.stdout = StringIO()
        output = sys.stdout.getvalue()

        self.assertIn("Epoch 1/1..", output)
        self.assertIn("Train loss:", output)
        self.assertIn("Validation loss:", output)
        self.assertIn("Validation accuracy:", output)

        sys.stdout = original_stdout


    def test_save_checkpoint(self):
        """Test the save_checkpoint function."""
        model = models.vgg16(pretrained=True)

        temp_dir = tempfile.mkdtemp()

        try:
            self.train_functions.save_checkpoint(model, save_dir=temp_dir, arch="vgg16", hidden_units=512, class_to_idx={})

            checkpoint_path = os.path.join(temp_dir, "checkpoint.pth")
            self.assertTrue(os.path.exists(checkpoint_path), f"Checkpoint file not found at {checkpoint_path}")
        finally:
            os.rmdir(temp_dir)


if __name__ == '__main__':
    unittest.main()
