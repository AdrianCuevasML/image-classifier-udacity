import json
import torch
from torchvision import transforms
from PIL import Image
import torchvision

class PredictFunctions:
    """
    A class containing functions for model prediction.
    """

    def __init__(self) -> None:
        """
        Initializes an instance of PredictFunctions.
        """
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a model checkpoint from the specified path.

        Parameters:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            torch.nn.Module: Loaded model.
        """
        checkpoint = torch.load(checkpoint_path)

        if checkpoint["arch"] == "vgg16":
            model = torchvision.models.vgg16(weights="DEFAULT")
        elif checkpoint["arch"] == "resnet18":
            model = torchvision.models.resnet18(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = checkpoint["classifier"]
        model.load_state_dict(checkpoint["state_dict"])
        model.class_to_idx = checkpoint["class_to_idx"]

        return model

    def predict(self, *, image_path: str, model, topk: int, category_names_path: str, gpu):
        """
        Make predictions using the specified model on the given image.

        Parameters:
            image_path (str): Path to the input image.
            model (torch.nn.Module): Trained model.
            topk (int): Return top K most likely classes.
            category_names_path (str): Path to category names mapping file.
            gpu (bool): Use GPU for inference.

        Returns:
            tuple: A tuple containing a list of top probabilities and a list of predicted class names.
        """
        category_to_name = self._open_category_to_names(category_names_path)
        model.eval()

        if gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model.to(device)
        image = self._process_image(image_path)
        image = image.unsqueeze(0)
        image = image.to(device)
        model.eval()

        top_ps, predicted_flowers = self._get_predicted_flowers(model, image, category_to_name, topk)

        return top_ps.tolist()[0], predicted_flowers

    def show_results(self, class_names, top_list):
        """
        Display the predicted class names and corresponding probabilities.

        Parameters:
            class_names (list): List of predicted class names.
            top_list (list): List of corresponding probabilities.
        """
        for i in range(len(class_names)):
            print(f"Class: {class_names[i]}, Probability: {top_list[i]:.3f}")

    def _open_category_to_names(self, category_names_path: str) -> dict:
        """
        Open and read the category-to-name mapping from a JSON file.

        Parameters:
            category_names_path (str): Path to category names mapping file.

        Returns:
            dict: Mapping of category indices to class names.
        """
        with open(category_names_path, "r") as f:
            category_to_name = json.load(f)

        return category_to_name

    def _process_image(self, image_path: str):
        """
        Preprocess the input image for model inference.

        Parameters:
            image_path (str): Path to the input image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        img_pil = Image.open(image_path)
        img_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        image = img_transforms(img_pil)
        return image

    def _get_predicted_flowers(self, model, image, topk: int, category_to_name):
        """
        Get the top K predicted classes and corresponding probabilities.

        Parameters:
            model (torch.nn.Module): Trained model.
            image (torch.Tensor): Preprocessed input image tensor.
            topk (int): Return top K most likely classes.
            category_to_name (dict): Mapping of category indices to class names.

        Returns:
            tuple: A tuple containing a tensor of top probabilities and a list of predicted class names.
        """
        with torch.no_grad():
            predicted = torch.exp(model(image))
        idx_to_flower = {
            index: category_to_name[name] for name, index in model.class_to_idx.items()
        }
        top_ps, top_classes = predicted.topk(topk, dim=1)
        predicted_flowers = [idx_to_flower[i] for i in top_classes.tolist()[0]]

        return top_ps, predicted_flowers
