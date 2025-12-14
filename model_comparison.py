# model_comparison.py
import tensorflow as tf
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import h5py

print("="*60)
print("PATH DEBUGGING")
print("="*60)
print(f"Current directory: {os.getcwd()}")
print(f"TensorFlow version: {tf.__version__}")
print("="*60 + "\n")

# --------------------------------------------------------------------
# FIXED LOADER — always use this to avoid 'batch_shape' errors
# --------------------------------------------------------------------


def load_model_with_fix(model_path):
    """Loads model safely even if H5 contains invalid 'batch_shape'."""
    print(f"\n📌 Loading model with compatibility fix: {model_path}")

    try:
        # ---- Try normal loading first ----
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("   ✓ Loaded normally (no fix needed)")
            return model
        except Exception as e:
            print("   ⚠ Normal load failed, applying batch_shape fix...")

        # ---- Fix for 'batch_shape' in config ----
        with h5py.File(model_path, "r") as f:
            config = f.attrs.get("model_config")
            if isinstance(config, bytes):
                config = config.decode("utf-8")

            config_json = json.loads(config)

            # Recursively replace batch_shape → batch_input_shape
            def fix(config_dict):
                if isinstance(config_dict, dict):
                    if "batch_shape" in config_dict:
                        config_dict["batch_input_shape"] = config_dict.pop("batch_shape")
                    for k in config_dict.values():
                        fix(k)
                elif isinstance(config_dict, list):
                    for i in config_dict:
                        fix(i)

            fix(config_json)

        # ---- Rebuild & load weights ----
        print("   🔧 Rebuilding model from fixed config...")
        model = tf.keras.models.model_from_json(json.dumps(config_json))
        model.load_weights(model_path)
        print("   ✓ Model fixed & weights loaded successfully")

        return model

    except Exception as e:
        print(f"\n❌ Critical load failure: {e}")
        raise






# --------------------------------------------------------------------
# COMPARATOR CLASS
# --------------------------------------------------------------------
class ModelComparator:
    def __init__(self):
        self.models = {}
        self.results = {}

    def add_model(self, name, model):
        self.models[name] = model
        print(f"✅ Added model '{name}'")

    def benchmark_inference_speed(self, model, input_shape, num_iterations=50):
        dummy = np.random.rand(1, *input_shape).astype(np.float32)

        for _ in range(5):
            model.predict(dummy, verbose=0)

        start = time.time()
        for _ in range(num_iterations):
            model.predict(dummy, verbose=0)
        end = time.time()

        avg_ms = (end - start) / num_iterations * 1000
        fps = 1000 / avg_ms
        return avg_ms, fps

    def compare_models(self, test_data, input_shape):
        for name, model in self.models.items():
            print("\n" + "="*80)
            print(f"Evaluating: {name}")
            print("="*80)

            res = model.evaluate(test_data, verbose=0)
            loss = res[0]
            acc = res[1] if len(res) > 1 else 0

            print(f" Loss: {loss:.4f}")
            print(f" Accuracy: {acc*100:.2f}%")

            print("\n⏱ Benchmarking speed...")
            avg_time, fps = self.benchmark_inference_speed(model, input_shape)
            print(f" Inference time: {avg_time:.2f} ms")
            print(f" FPS: {fps:.2f}")

            size = 0
            try:
                temp = f"{name.replace(' ', '_')}_temp.h5"
                model.save(temp)
                size = os.path.getsize(temp) / (1024 * 1024)
                os.remove(temp)
            except:
                pass

            params = model.count_params()

            self.results[name] = {
                "accuracy": acc * 100,
                "inference_time_ms": avg_time,
                "fps": fps,
                "model_size_mb": size,
                "parameters": params,
            }

        return pd.DataFrame(self.results).T


# --------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------
if __name__ == "__main__":
    print(tf.__version__)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")

    cnn_h5 = os.path.join(models_dir, "final_cnn_model.h5")
    mobile_h5 = os.path.join(models_dir, "final_mobilenet_model.h5")

    print("\nLooking for model files:")
    print(f" CNN: {cnn_h5} ({'OK' if os.path.exists(cnn_h5) else 'MISSING'})")
    print(f" MobileNet: {mobile_h5} ({'OK' if os.path.exists(mobile_h5) else 'MISSING'})\n")

    comparator = ModelComparator()

    # Load models using FIXED loader
    print("\n========= LOADING MODELS ========")
    cnn_model = load_model_with_fix(cnn_h5)
    cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    mobilenet_model = load_model_with_fix(mobile_h5)
    mobilenet_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    comparator.add_model("Custom CNN", cnn_model)
    comparator.add_model("MobileNetV2", mobilenet_model)

    # Load test data
    print("\n========= LOADING TEST DATA ========")
    test_path = os.path.join(script_dir, "dataset", "val")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset missing: {test_path}")

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_path, target_size=(224, 224), batch_size=32, class_mode="categorical", shuffle=False
    )

    print(f"\nLoaded {test_gen.samples} test samples across {test_gen.num_classes} classes")

    print("\n========= STARTING COMPARISON ========")
    results_df = comparator.compare_models(test_gen, (224, 224, 3))

    print("\n===== RESULTS =====")
    print(results_df)

    results_df.to_csv("model_comparison_results.csv")
    print("\nSaved: model_comparison_results.csv")
