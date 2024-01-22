import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
import torch
from torchvision import models
from PIL import Image
from tempfile import NamedTemporaryFile
from image_classifier.predict_functions import PredictFunctions

class TestPredictFunctions(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment.
        """
        self.predict_functions = PredictFunctions()
        self.temp_file = NamedTemporaryFile(delete=False)

    def tearDown(self):
        """
        Tear down the test environment.
        """
        self.temp_file.close()

    @patch('torchvision.models.vgg16')
    def test_load_checkpoint_vgg16(self, mock_vgg16):
        """
        Test loading a VGG16 model checkpoint.
        """
        mock_model = MagicMock()
        mock_vgg16.return_value = mock_model

        sample_model = models.vgg16(pretrained=False)
        sample_model.classifier = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 1000)
        )
        sample_model_dict = {
            "arch": "vgg16",
            "classifier": sample_model.classifier,
            "state_dict": sample_model.state_dict(),
            "class_to_idx": {"class1": 0, "class2": 1}
        }
        torch.save(sample_model_dict, self.temp_file.name)

        loaded_model = self.predict_functions.load_checkpoint(self.temp_file.name)
        mock_vgg16.assert_called_once_with(weights="DEFAULT")

        self.assertIs(loaded_model, mock_model)

    @patch('image_classifier.predict_functions.YourClass._open_category_to_names')
    @patch('image_classifier.predict_functions.YourClass._process_image')
    @patch('image_classifier.predict_functions.YourClass._get_predicted_flowers')
    def test_predict(self, mock_get_predicted_flowers, mock_process_image, mock_open_category_to_names):
        """
        Test the prediction process.
        """
        mock_open_category_to_names.return_value = {"0": "class1", "1": "class2"}
        mock_process_image.return_value = torch.rand((3, 224, 224))
        mock_model = MagicMock()
        mock_get_predicted_flowers.return_value = (torch.Tensor([0.8, 0.2]), ["class1", "class2"])

        top_probs, predicted_classes = self.predict_functions.predict(
            image_path="fake_path",
            model=mock_model,
            topk=2,
            category_names_path="fake_path",
            gpu=True
        )

        mock_open_category_to_names.assert_called_once_with("fake_path")
        mock_process_image.assert_called_once_with("fake_path")
        mock_get_predicted_flowers.assert_called_once_with(mock_model, torch.rand((3, 224, 224)),
                                                           2, {"0": "class1", "1": "class2"})

        self.assertEqual(top_probs, [0.8, 0.2])
        self.assertEqual(predicted_classes, ["class1", "class2"])

    def test_show_results(self):
        """
        Test displaying predicted results.
        """
        class_names = ["class1", "class2"]
        top_list = [0.8, 0.2]

        # Capture standard output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            self.predict_functions.show_results(class_names, top_list)

        output = mock_stdout.getvalue().strip()

        expected_output = "Class: class1, Probability: 0.800\nClass: class2, Probability: 0.200"
        self.assertEqual(output, expected_output)

if __name__ == '__main__':
    unittest.main()
