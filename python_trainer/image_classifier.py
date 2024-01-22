from train_functions import TrainFunctions
from predict_functions import PredictFunctions

class ImageClassifierTrainer:

    def __init__(self,
        train_functions: TrainFunctions,
        data_directory: str,
        save_dir: str,
        architecture: str,
        learning_rate: float,
        hidden_units: int,
        epochs: int,
        gpu,
    ) -> None:
        self.train_functions = train_functions
        self.data_directory = data_directory
        self. save_dir = save_dir
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.gpu = gpu

    def train_image_classifier(self):
        train_loader, valid_loader, test_loader, train_data = self.train_functions.load_data(self.data_directory)
        model = self.train_functions.create_model(self.architecture, self.hidden_units)
        self.train_functions.train_model(
            model, train_loader, valid_loader, self.learning_rate, self.epochs, self.gpu
        )
        self.train_functions.save_checkpoint(
            model, self.save_dir, self.architecture, self.hidden_units, train_data.class_to_idx
        )

class ImageClassifierPredicter:
    
    def __init__(self,
        predict_functions: PredictFunctions,
        image_path: str,
        checkpoint_path: str,
        top_k: int,
        category_names_path: str,
        gpu_predict,
    ) -> None:
        self.predict_functions = predict_functions
        self.image_path = image_path
        self.checkpoint_path = checkpoint_path
        self.top_k = top_k
        self.category_names_path = category_names_path
        self.gpu_predict = gpu_predict

    def predict_image_classifier(self) -> None:
        
        model = self.predict_functions.load_checkpoint(self.checkpoint_path)
        top_list, class_names = self.predict_functions.predict(
            image_path=self.image_path, model=model, topk=self.top_k, 
            category_names_path=self.category_names_path, gpu=self.gpu_predict
            )
        self.predict_functions.show_results(class_names=class_names, top_list=top_list)