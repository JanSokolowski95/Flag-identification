import torchmetrics
import torch
import torchvision
import lightning as L


class Classifier(L.LightningModule):
    """Classifier model for the flags dataset.

    Attributes:
        model:
            ResNet34 model with a custom prediction head.
        loss_fn:
            CrossEntropyLoss function.
        accuracy:
            Accuracy metric.

    """

    def __init__(self, num_classes: int = 194):
        """Initializes the model.

        Args:
            num_classes:
                Number of classes in the dataset. Defaults to 194.

        """
        super().__init__()
        model = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.DEFAULT
        )
        model.fc = torch.nn.LazyLinear(194)
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def _step(self, batch, label: str) -> torch.Tensor:
        """Runs a step of the model.

        Convenience method to keep the code DRY.

        Args:
            batch:
                A batch of data.
            label:
                Label for the step.

        Returns:
            Loss for the step.

        """
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        self.log(f"{label}_loss", loss)
        self.log(f"{label}_accuracy", self.accuracy(outputs, targets))
        return loss

    def training_step(self, batch):
        """Runs a training step of the model.

        Args:
            batch:
                A batch of data.

        Returns:
            Loss for the step.

        """
        return self._step(batch, "train")

    def validation_step(self, batch):
        """Runs a validation step of the model.

        Args:
            batch:
                A batch of data.

        """
        self._step(batch, "val")

    def test_step(self, batch):
        """Runs a test step of the model.

        Args:
            batch:
                A batch of data.
        """
        self._step(batch, "test")

    def configure_optimizers(self):
        """Configures the optimizer for the model.

        Returns:
            AdamW optimizer.

        """
        return torch.optim.AdamW(self.model.parameters(), lr=3e-4)
