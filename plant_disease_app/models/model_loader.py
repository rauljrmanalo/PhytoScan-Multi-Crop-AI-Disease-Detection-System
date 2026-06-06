"""
models/model_loader.py
═══════════════════════════════════════════════════════════════════════
Loads trained VGG16 models and runs inference for PhytoScan.

Rice model targets the Kaggle dataset:
    loki4514/rice-leaf-diseases-detection
    4 classes: Bacterial Blight | Blast | Brown Spot | Tungro
"""

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf


# ── Preprocessing ─────────────────────────────────────────────────────
IMG_SIZE = (224, 224)

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image and normalise it exactly as done during training.
    Training used ImageDataGenerator(rescale=1./255), so we divide by 255 here.
    Input shape expected by VGG16: (1, 224, 224, 3), float32, values in [0, 1].
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # → (1, 224, 224, 3)


# ── Classifier ─────────────────────────────────────────────────────────
class VGG16Classifier:
    """
    Wraps one trained VGG16 .keras (or .h5) model for a single crop.

    Attributes
    ----------
    model_path       : path to rice_vgg16.keras
    class_index_path : path to rice_class_indices.json
                       Format produced by train_rice_vgg16.py:
                       {"0": "bacterial_blight", "1": "rice_blast",
                        "2": "brown_spot",        "3": "tungro"}
    """

    def __init__(self, model_path: str, class_index_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(class_index_path):
            raise FileNotFoundError(f"Class index JSON not found: {class_index_path}")

        print(f"  Loading: {model_path}")
        self.model = tf.keras.models.load_model(model_path)

        with open(class_index_path) as f:
            raw = json.load(f)
        # Ensure keys are ints: {0: "bacterial_blight", 1: "rice_blast", ...}
        self.class_index = {int(k): v for k, v in raw.items()}
        print(f"  Class map: {self.class_index}")

    def predict(self, image_path: str) -> dict:
        """
        Run inference on a single image.

        Returns
        -------
        {
          "disease_id"       : str    — matches DISEASE_DB key in app.py
          "confidence"       : float  — 0–100 (highest softmax probability)
          "affected_area_pct": float  — pixel-level lesion estimate 0–90
          "top3"             : list   — top 3 predictions with confidence
        }
        """
        arr   = preprocess_image(image_path)
        probs = self.model.predict(arr, verbose=0)[0]   # (num_classes,)

        top_idx    = int(np.argmax(probs))
        confidence = float(probs[top_idx]) * 100
        disease_id = self.class_index.get(top_idx, "unknown")

        affected_area = (
            self._estimate_affected_area(image_path)
            if disease_id not in ("healthy", "unknown")
            else 0.0
        )

        return {
            "disease_id":        disease_id,
            "confidence":        round(confidence, 1),
            "affected_area_pct": round(affected_area, 1),
            "top3":              self._top3(probs),
        }

    def _top3(self, probs: np.ndarray) -> list:
        indices = np.argsort(probs)[::-1][:3]
        return [
            {
                "disease_id": self.class_index.get(int(i), "unknown"),
                "confidence": round(float(probs[i]) * 100, 1)
            }
            for i in indices
        ]

    def _estimate_affected_area(self, image_path: str) -> float:
        """
        Fast pixel heuristic — estimates what % of the leaf shows
        disease-related discolouration (brown, yellow, dark necrotic).

        Replace with a segmentation model output for precise results.
        """
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Brown / tan lesions: high R, moderate G, low B, R dominant
        brown = (r > 120) & (g > 60) & (b < 100) & (r > g * 1.15)

        # Yellow / orange (Tungro, early blight): high R+G, low B
        yellow = (r > 160) & (g > 140) & (b < 80)

        # Dark necrotic tissue: all channels low
        dark = (r < 75) & (g < 75) & (b < 75)

        mask = brown | yellow | dark
        pct = float(mask.sum()) / mask.size * 100
        return min(round(pct, 1), 90.0)


# ── Model Registry ─────────────────────────────────────────────────────
# Point each crop to its trained model and class-index JSON.
# Paths are relative to this file's parent directory.

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'weights')

REGISTRY_CONFIG = {
    # ── Rice: trained on loki4514/rice-leaf-diseases-detection ──────
    "rice": {
        "model": os.path.join(MODELS_DIR, "rice_vgg16.keras"),
        "index": os.path.join(MODELS_DIR, "rice_class_indices.json"),
    },
    # ── Other crops: add their model files when ready ────────────────
    "corn": {
        "model": os.path.join(MODELS_DIR, "corn_vgg16.keras"),
        "index": os.path.join(MODELS_DIR, "corn_class_indices.json"),
    },
    "banana": {
        "model": os.path.join(MODELS_DIR, "banana_vgg16.keras"),
        "index": os.path.join(MODELS_DIR, "banana_class_indices.json"),
    },
    "chilli": {
        "model": os.path.join(MODELS_DIR, "chilli_vgg16.keras"),
        "index": os.path.join(MODELS_DIR, "chilli_class_indices.json"),
    },
    "onion": {
        "model": os.path.join(MODELS_DIR, "onion_vgg16.keras"),
        "index": os.path.join(MODELS_DIR, "onion_class_indices.json"),
    },
}

_loaded: dict = {}


def load_all_models():
    """
    Pre-load all available models at Flask startup.
    Crops with missing model files are silently skipped and
    will fall back to mock_classify() in app.py.
    """
    print("\n[PhytoScan] Loading VGG16 models...")
    for crop, cfg in REGISTRY_CONFIG.items():
        model_exists = os.path.exists(cfg["model"])
        index_exists = os.path.exists(cfg["index"])
        if model_exists and index_exists:
            try:
                _loaded[crop] = VGG16Classifier(cfg["model"], cfg["index"])
                print(f"  ✅ {crop:8s} — VGG16 model loaded")
            except Exception as exc:
                print(f"  ❌ {crop:8s} — Failed to load: {exc}")
        else:
            missing = []
            if not model_exists: missing.append(os.path.basename(cfg["model"]))
            if not index_exists: missing.append(os.path.basename(cfg["index"]))
            print(f"  ⏭️  {crop:8s} — mock classifier (missing: {', '.join(missing)})")
    print(f"[PhytoScan] {len(_loaded)}/{len(REGISTRY_CONFIG)} models ready.\n")


def get_classifier(crop: str):
    """Return the loaded VGG16Classifier for a crop, or None (→ mock fallback)."""
    return _loaded.get(crop)
