import torchmetrics
import torch
from torchvision import transforms, datasets
import torchvision
import lightning as L


class Classifier(L.LightningModule):
    def __init__(self, num_classes: int = 194):
        super().__init__()
        model = torchvision.models.resnet34(pretrained=True)
        model.fc = torch.nn.LazyLinear(194)
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def _step(self, batch, label: str) -> torch.Tensor:
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        self.log(f"{label}_loss", loss)
        self.log(f"{label}_accuracy", self.accuracy(outputs, targets))
        return loss

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        self._step(batch, "val")

    def test_step(self, batch):
        self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=3e-4)
