import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from features import extract_enhanced_features, recompress_jpeg


def run_compression_check(raw_images, idx_test, y_test, trained,
                          qualities=(95, 75, 50)):
    rows = []
    for q in qualities:
        X_enh_q = np.asarray([
            extract_enhanced_features(recompress_jpeg(raw_images[i], q))
            for i in idx_test
        ], dtype=np.float32)

        for name, (model, p_orig) in trained.items():
            preds = model.predict(X_enh_q)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            orig_acc = accuracy_score(y_test, p_orig)
            rows.append({
                "Model": name,
                "JPEG Quality": q,
                "Accuracy": acc,
                "Delta vs. Original": acc - orig_acc,
                "F1": f1,
            })
    return rows


def print_report(rows):
    print("\n--- Compression Shortcut Diagnostic Report ---")
    print(f"{'Model':<16}{'Q':>5}{'Acc':>10}{'Delta':>10}{'F1':>10}")
    for r in rows:
        print(f"{r['Model']:<16}{r['JPEG Quality']:>5}"
              f"{r['Accuracy']:>10.4f}{r['Delta vs. Original']:>+10.4f}"
              f"{r['F1']:>10.4f}")