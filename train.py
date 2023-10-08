import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader as Loader


def load_data(data_dir):
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    valid_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.RandomRotation(40),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    batch_size = 64

    train_loader = Loader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = Loader(valid_data, batch_size=batch_size)
    test_loader = Loader(test_data, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, train_data


def create_model(arch, hidden_units):
    if arch == "vgg16":
        model = models.vgg16(weights="DEFAULT")
    elif arch == "resnet18":
        model = models.resnet18(weights="DEFAULT")
    else:
        raise ValueError(
            "Architecture not supported. Please choose 'vgg16' or 'resnet18'."
        )

    for param in model.parameters():
        param.requires_grad = False

    if arch == "vgg16":
        classifier = nn.Sequential(
            nn.Linear(25088, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1),
        )
    elif arch == "resnet18":
        classifier = nn.Sequential(
            nn.Linear(512, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1),
        )

    model.classifier = classifier
    return model


def train_model(model, train_loader, valid_loader, learning_rate, epochs, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0
            model.eval()

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels)

                    valid_loss += batch_loss.item()

                    ps = torch.exp(outputs)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(
                f"Epoch {epoch + 1}/{epochs}.. "
                f"Train loss: {running_loss / len(train_loader):.3f}.. "
                f"Validation loss: {valid_loss / len(valid_loader):.3f}.. "
                f"Validation accuracy: {accuracy / len(valid_loader):.3f}"
            )


def save_checkpoint(model, save_dir, arch, hidden_units, class_to_idx):
    checkpoint = {
        "arch": arch,
        "hidden_units": hidden_units,
        "class_to_idx": class_to_idx,
        "classifier": model.classifier,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, save_dir)
    print(f"Checkpoint saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a deep learning model on a dataset."
    )
    parser.add_argument("data_directory", help="Path to the data directory")
    parser.add_argument(
        "--save_dir", default="checkpoint.pth", help="Directory to save checkpoints"
    )
    parser.add_argument("--arch", default="vgg16", help="Architecture (vgg16)")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--hidden_units", type=int, default=512, help="Number of hidden units"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()

    train_loader, valid_loader, test_loader, train_data = load_data(args.data_directory)
    model = create_model(args.arch, args.hidden_units)
    train_model(
        model, train_loader, valid_loader, args.learning_rate, args.epochs, args.gpu
    )
    save_checkpoint(
        model, args.save_dir, args.arch, args.hidden_units, train_data.class_to_idx
    )


if __name__ == "__main__":
    main()
