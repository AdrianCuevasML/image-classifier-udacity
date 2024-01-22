# modules
from image_classifier import ImageClassifierTrainer, ImageClassifierPredicter
from train_functions import TrainFunctions
from predict_functions import PredictFunctions
# locals
import argparse

def main():
    """
    Main function to initiate training or prediction based on command line arguments.
    """
    args = initializer()
    print(args)

    if args[0] == "train":
        (mode, data_directory, save_dir, architecture,
         learning_rate, hidden_units, epochs, gpu
        ) = initializer()

        image_classifier_trainer = ImageClassifierTrainer(
            train_functions=TrainFunctions(),
            data_directory=data_directory,
            save_dir=save_dir,
            architecture=architecture,
            learning_rate=learning_rate,
            hidden_units=hidden_units,
            epochs=epochs,
            gpu=gpu,
        )

        image_classifier_trainer.train_image_classifier()

    elif args[0] == "predict":
        (mode, image_path, checkpoint_path,
         top_k, category_names_path, gpu_predict
        ) = initializer()

        image_classifier_predicter = ImageClassifierPredicter(
            predict_functions=PredictFunctions,
            image_path=image_path,
            checkpoint_path=checkpoint_path,
            top_k=top_k,
            category_names_path=category_names_path,
            gpu_predict=gpu_predict,
        )

        image_classifier_predicter.predict_image_classifier()

def initializer():
    """
    Function to parse command line arguments and return them as a tuple.
    Returns:
        tuple: A tuple containing the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train and predict deep learning model on a dataset.")

    # Argument to select mode (train or predict)
    parser.add_argument("--mode", choices=["train", "predict"], help="Mode: train or predict")
    args, unknown_args = parser.parse_known_args()

    if args.mode == "train":
        train_group = parser.add_argument_group("Training Arguments")
        # Arguments for model training
        train_group.add_argument("--data_directory", help="Path to the data directory")
        train_group.add_argument("--save_dir", default="checkpoint.pth", help="Directory to save checkpoint")
        train_group.add_argument("--architecture", default="vgg16", help="Architecture (vgg16)")
        train_group.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
        train_group.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
        train_group.add_argument("--epochs", type=int, default=20, help="Number of epochs")
        train_group.add_argument("--gpu", action="store_true", help="Use GPU for training")
        args = parser.parse_args()

        return(args.mode, args.data_directory, args.save_dir, args.architecture,
               args.learning_rate, args.hidden_units, args.epochs, args.gpu)

    elif args.mode == "predict":
        predict_group = parser.add_argument_group("Prediction Arguments")
        # Arguments for model prediction
        predict_group.add_argument("--image_path", help="Path to the input image")
        predict_group.add_argument("--checkpoint_path", help="Path to the checkpoint file")
        predict_group.add_argument("--top_k", type=int, default=1, help="Return top K most likely classes")
        predict_group.add_argument("--category_names_path", default="category_to_name.json", help="Path to category names mapping file")
        predict_group.add_argument("--gpu_predict", action="store_true", help="Use GPU for inference in prediction mode")
        args = parser.parse_args()

        return(args.mode, args.image_path, args.checkpoint_path, args.top_k,
               args.category_names_path, args.gpu_predict)

    else:
        print("Error: Choose 'train' or 'predict' mode")

if __name__ == "__main__":
    main()