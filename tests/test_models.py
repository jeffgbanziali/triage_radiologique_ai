"""Tests unitaires pour les architectures du projet. Usage : pytest tests/test_models.py -v"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Fixtures
@pytest.fixture
def dummy_batch():
    """Batch factice de 4 images RGB 64x64."""
    return torch.randn(4, 3, 64, 64)


@pytest.fixture
def dummy_labels():
    return torch.randint(0, 2, (4, 14)).float()


# Classification
class TestCNNFromScratch:
    def test_output_shape(self, dummy_batch):
        from src.models.cnn_from_scratch import CNNFromScratch
        model = CNNFromScratch(num_classes=14, base_channels=16, num_blocks=4)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, 14), f"Shape attendue (4, 14), obtenu {out.shape}"

    def test_gradient_flows(self, dummy_batch, dummy_labels):
        from src.models.cnn_from_scratch import CNNFromScratch
        model = CNNFromScratch(num_classes=14, base_channels=16, num_blocks=4)
        logits = model(dummy_batch)
        loss = torch.nn.BCEWithLogitsLoss()(logits, dummy_labels)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Pas de gradient pour {name}"

    def test_resolution_224(self):
        from src.models.cnn_from_scratch import CNNFromScratch
        model = CNNFromScratch(num_classes=14, base_channels=16, num_blocks=4)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 14)


class TestDenseNet121:
    def test_output_shape_frozen(self, dummy_batch):
        from src.models.transfer_learning import DenseNet121Classifier
        model = DenseNet121Classifier(num_classes=14, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, 14)

    def test_freeze_unfreeze(self):
        from src.models.transfer_learning import DenseNet121Classifier
        model = DenseNet121Classifier(num_classes=14, pretrained=False)
        model.freeze_backbone()
        frozen_count = sum(1 for p in model.features.parameters() if not p.requires_grad)
        total_feat   = sum(1 for _ in model.features.parameters())
        assert frozen_count == total_feat, "Tous les params backbone doivent etre geles"
        model.unfreeze_last_block("denseblock4")
        unfrozen = sum(1 for p in model.features.parameters() if p.requires_grad)
        assert unfrozen > 0, "Au moins un param doit etre degele apres unfreeze"


class TestDeiTTiny:
    def test_output_shape(self, dummy_batch):
        from src.models.vit import DeiTTinyClassifier
        model = DeiTTinyClassifier(num_classes=14, img_size=64, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, 14)


# Anomalie
class TestAutoencoder:
    def test_ae_reconstruction_shape(self, dummy_batch):
        from src.anomaly.autoencoder import ConvAutoencoder
        model = ConvAutoencoder(in_channels=3, base_channels=16, latent_dim=64)
        model.eval()
        with torch.no_grad():
            x_hat, z = model(dummy_batch)
        assert x_hat.shape == dummy_batch.shape, "La reconstruction doit avoir la meme shape que l'input"
        assert z.shape == (4, 64)

    def test_vae_shapes(self, dummy_batch):
        from src.anomaly.autoencoder import ConvVAE
        model = ConvVAE(in_channels=3, base_channels=16, latent_dim=64)
        model.eval()
        with torch.no_grad():
            x_hat, mu, logvar = model(dummy_batch)
        assert x_hat.shape == dummy_batch.shape
        assert mu.shape == (4, 64)
        assert logvar.shape == (4, 64)

    def test_vae_loss(self, dummy_batch):
        from src.anomaly.autoencoder import ConvVAE
        model = ConvVAE(in_channels=3, base_channels=16, latent_dim=64)
        x_hat, mu, logvar = model(dummy_batch)
        loss, parts = model.loss(dummy_batch, x_hat, mu, logvar)
        assert loss.item() > 0
        assert "loss_rec" in parts
        assert "loss_kl" in parts


# Metriques
class TestMetrics:
    def test_compute_metrics_shape(self):
        from src.evaluation.metrics import compute_metrics
        import numpy as np
        y_true = np.random.randint(0, 2, (100, 14))
        y_prob = np.random.rand(100, 14)
        metrics = compute_metrics(y_true, y_prob)
        assert "f1_macro" in metrics
        assert "mcc_macro" in metrics
        assert "auc_macro" in metrics
        assert "balanced_accuracy" in metrics

    def test_perfect_classifier(self):
        from src.evaluation.metrics import compute_metrics
        import numpy as np
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_prob = y_true.astype(float)
        metrics = compute_metrics(y_true, y_prob, threshold=0.5)
        assert metrics["f1_macro"] == pytest.approx(1.0, abs=0.01)


# Data pipeline
class TestAugmentation:
    def test_train_transform_output_size(self):
        from src.data.augmentation import get_train_transforms
        from PIL import Image
        transform = get_train_transforms(resolution=64)
        img = Image.new("RGB", (256, 256))
        tensor = transform(img)
        assert tensor.shape == (3, 64, 64)

    def test_val_is_deterministic(self):
        from src.data.augmentation import get_val_transforms
        from PIL import Image
        transform = get_val_transforms(resolution=64)
        img = Image.new("RGB", (256, 256))
        t1, t2 = transform(img), transform(img)
        assert torch.allclose(t1, t2), "Le val transform doit etre deterministe"
