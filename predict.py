import argparse
import json
import torch
from torchvision import transforms, models
from PIL import Image
import torchvision


def load_checkpoint(checkpoint_path):
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


def process_image(image_path):
    img_pil = Image.open(image_path)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    image = img_transforms(img_pil)
    return image


def predict(image_path, model, topk, cat_to_name, gpu):
    model.eval()

    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        predicted = torch.exp(model(image))
    idx_to_flower = {
        index: cat_to_name[name] for name, index in model.class_to_idx.items()
    }
    top_ps, top_classes = predicted.topk(topk, dim=1)
    predicted_flowers = [idx_to_flower[i] for i in top_classes.tolist()[0]]

    return top_ps.tolist()[0], predicted_flowers


def main():
    parser = argparse.ArgumentParser(
        description="Predict flower name from an image using a trained model."
    )
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("checkpoint", help="Path to the checkpoint file")
    parser.add_argument(
        "--top_k", type=int, default=1, help="Return top K most likely classes"
    )
    parser.add_argument(
        "--category_names",
        default="cat_to_name.json",
        help="Path to category names mapping file",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    with open(args.category_names, "r") as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    top_p, class_names = predict(
        args.image_path, model, args.top_k, cat_to_name, args.gpu
    )

    for i in range(len(class_names)):
        print(f"Class: {class_names[i]}, Probability: {top_p[i]:.3f}")


if __name__ == "__main__":
    main()
