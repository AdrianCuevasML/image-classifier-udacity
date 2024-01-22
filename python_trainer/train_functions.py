import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader as Loader

class TrainFunctions:

    def __init__(self, data_dir: str) -> None:

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.batch_size = 64
        self.train_dir = data_dir + "/train"
        self.valid_dir = data_dir + "/valid"
        self.test_dir = data_dir + "/test"

    def load_data(self):

        train_transforms, valid_transforms, test_transform = self._transform_data()

        train_data = datasets.ImageFolder(self.train_dir, transform=train_transforms)
        valid_data = datasets.ImageFolder(self.valid_dir, transform=valid_transforms)
        test_data = datasets.ImageFolder(self.test_dir, transform=test_transform)

        train_loader = Loader(train_data, batch_size=self.batch_size, shuffle=True)
        valid_loader = Loader(valid_data, batch_size=self.batch_size)
        test_loader = Loader(test_data, batch_size=self.batch_size)

        return train_data, train_loader, valid_loader, test_loader

    def create_model(self, arch: str, hidden_units: int):

        if arch == "vgg16":
            model = self._vgg16_architecture_configuration(hidden_units)
        elif arch == "resnet18":
            model = self._resnet18_architecture_configuration(hidden_units)
        else:
            raise ValueError(
                "Architecture not supported. Please choose 'vgg16' or 'resnet18'."
            )
    
        return model

    def train_model(self, model, train_loader, valid_loader, learning_rate: float, epochs: int, gpu) -> None:

        device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            train_loss = self._train_one_epoch(model, train_loader, optimizer, criterion, device)
            valid_loss, accuracy = self._validate(model, valid_loader, criterion, device)

            print(
                f"Epoch {epoch + 1}/{epochs}.. "
                f"Train loss: {train_loss:.3f}.. "
                f"Validation loss: {valid_loss:.3f}.. "
                f"Validation accuracy: {accuracy:.3f}"
            )


    def save_checkpoint(model, save_dir: str, arch: str, hidden_units: int, class_to_idx):

        checkpoint = {
            "arch": arch,
            "hidden_units": hidden_units,
            "class_to_idx": class_to_idx,
            "classifier": model.classifier,
            "state_dict": model.state_dict(),
        }
        torch.save(checkpoint, save_dir)
        print(f"Checkpoint saved to {save_dir}")

    def _transform_data(self):

        train_transforms = transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        valid_transforms = transforms.Compose(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.RandomRotation(40),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )    

        return train_transforms, valid_transforms, test_transform
    
    def _vgg16_architecture_configuration(self, hidden_units: int):

        model = models.vgg16(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False

        classifier = torch.nn.Sequential(
            torch.nn.Linear(25088, hidden_units),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_units, 102),
            torch.nn.LogSoftmax(dim=1),
        )
        model.classifier = classifier

        return model
    def _resnet18_architecture_configuration(self, hidden_units: int):

        model = models.resnet18(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False

        classifier = torch.nn.Sequential(
            torch.nn.Linear(512, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_units, 102),
            torch.nn.LogSoftmax(dim=1),
        )
        model.classifier = classifier

        return model

    def _train_one_epoch(self, model, train_loader, optimizer, criterion, device):

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

        return running_loss / len(train_loader)

    def _validate(self, model, valid_loader, criterion, device):

        model.eval()
        valid_loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)

                valid_loss += batch_loss.item()

                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += equals.mean().item()

        return valid_loss / len(valid_loader), accuracy / len(valid_loader)
    