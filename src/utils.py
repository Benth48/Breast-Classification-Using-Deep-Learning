import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import math

# GradCAM++ heatmap for a single image

def _build_split_models(model, last_conv_layer_name=None):
    # Detect nested backbone
    backbone = next(
        (l for l in model.layers if isinstance(l, tf.keras.Model)),
        None
    )

    if backbone is not None:
        # NESTED PATH (ResNet50) 
        if last_conv_layer_name is None:
            # Default: backbone's own output layer.
            # For ResNet50(include_top=False) this is conv5_block3_out —
            # the last residual block's ReLU output, richest feature map,
            # and a clean split point (no skip connections after it).
            target_name = backbone.layers[-1].name
        else:
            target_name = last_conv_layer_name

        target     = backbone.get_layer(target_name)
        conv_model = tf.keras.Model(backbone.inputs, target.output,
                                    name="conv_model")

        # classifier: new Input(conv_shape) → head layers (GAP, Dense, …)
        cnn_inp = tf.keras.Input(shape=target.output.shape[1:], name="conv_input")
        x = cnn_inp
        for layer in model.layers:
            # skip InputLayer and the backbone itself; apply everything else
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            if layer is backbone:
                continue
            x = layer(x)
        classifier_model = tf.keras.Model(cnn_inp, x, name="classifier_model")

    else:
        # FLAT PATH (ShallowCNN Sequential)
        if last_conv_layer_name is None:
            # Scan model.layers directly (all layers are visible here)
            last_conv_layer_name = next(
                (l.name for l in reversed(model.layers)
                 if isinstance(l, tf.keras.layers.Conv2D)),
                None
            )
            if last_conv_layer_name is None:
                raise ValueError("No Conv2D layer found in the model.")

        target     = model.get_layer(last_conv_layer_name)
        conv_model = tf.keras.Model(model.inputs, target.output,
                                    name="conv_model")

        # classifier: iterate everything that comes after the target layer
        cnn_inp = tf.keras.Input(shape=target.output.shape[1:], name="conv_input")
        x = cnn_inp
        found = False
        for layer in model.layers:
            if found:
                x = layer(x)
            if layer.name == last_conv_layer_name:
                found = True
        classifier_model = tf.keras.Model(cnn_inp, x, name="classifier_model")

    return conv_model, classifier_model


def get_gradcam_pp(model, img_array, class_idx,
                   last_conv_layer_name=None,
                   _split_cache={}):
    # Cache split models so they are built only once per (model, layer) pair
    cache_key = (id(model), last_conv_layer_name)
    if cache_key not in _split_cache:
        _split_cache[cache_key] = _build_split_models(model, last_conv_layer_name)
    conv_model, classifier_model = _split_cache[cache_key]

    img_4d = tf.cast(img_array[np.newaxis, ...], tf.float32)   # (1, H, W, 3)

    # Run conv_model once (no gradient needed here)
    conv_out_val = conv_model(img_4d, training=False)           # (1, h, w, C)
    conv_var = tf.Variable(conv_out_val)

    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            preds = classifier_model(conv_var, training=False)  # (1, num_cls)
            score = preds[:, class_idx]                         # scalar
        grads1 = tape1.gradient(score, conv_var)                # ∂score/∂A
    grads2 = tape2.gradient(grads1, conv_var)                   # ∂²score/∂A²

    conv_np = conv_out_val.numpy()[0]    # (h, w, C)
    g1      = grads1.numpy()[0]          # (h, w, C)
    g2      = grads2.numpy()[0]          # (h, w, C)

    # GradCAM++ alpha weights:  α = g2 / (2·g2 + Σ_xy(A · g3))
    # g3 ≈ g2·g1  (efficient 3rd-order approximation)
    g3  = g2 * g1 + 1e-8
    num = g2
    den = 2.0 * g2 + np.sum(conv_np * g3, axis=(0, 1), keepdims=True)
    den = np.where(np.abs(den) < 1e-8, 1e-8, den)

    alpha   = num / den                                          # (h, w, C)
    weights = np.sum(alpha * np.maximum(g1, 0), axis=(0, 1))   # (C,)

    cam = np.sum(conv_np * weights, axis=-1)                    # (h, w)
    cam = np.maximum(cam, 0)                                    # ReLU
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.astype(np.float32)


def overlay_heatmap(img_array, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    h, w = img_array.shape[:2]
    heat_uint8 = np.uint8(255 * heatmap)
    heat_resized = cv2.resize(heat_uint8, (w, h))
    heat_color = cv2.applyColorMap(heat_resized, colormap)           # BGR
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)         # → RGB

    img_uint8 = np.uint8(255 * img_array)
    blended = cv2.addWeighted(img_uint8, 1 - alpha, heat_color, alpha, 0)
    return blended

# Collect misclassified samples from a generator

def collect_misclassified(model, generator, class_names, max_per_class=5):
    generator.reset()
    all_imgs, all_true = [], []

    for _ in range(len(generator)):
        imgs, labels = next(generator)
        all_imgs.append(imgs)
        all_true.extend(np.argmax(labels, axis=1))

    all_imgs = np.concatenate(all_imgs, axis=0)[:generator.samples]
    all_true = np.array(all_true)[:generator.samples]

    preds = model.predict(all_imgs, verbose=0)
    all_pred = np.argmax(preds, axis=1)
    confidence = np.max(preds, axis=1)

    misclassified = []
    per_class_count = {c: 0 for c in range(len(class_names))}

    wrong_mask = all_true != all_pred
    wrong_indices = np.where(wrong_mask)[0]

    # Shuffle so we get variety
    rng = np.random.default_rng(42)
    rng.shuffle(wrong_indices)

    for idx in wrong_indices:
        true_idx = all_true[idx]
        pred_idx = all_pred[idx]
        if per_class_count[true_idx] >= max_per_class:
            continue
        per_class_count[true_idx] += 1
        misclassified.append({
            "img":        all_imgs[idx],         # preprocessed — used for GradCAM gradients
            "img_display": np.clip(              # rescaled back to [0,1] for display
                (all_imgs[idx] - all_imgs[idx].min()) /
                (all_imgs[idx].max() - all_imgs[idx].min() + 1e-8),
                0, 1
            ).astype(np.float32),
            "true_idx":   int(true_idx),
            "pred_idx":   int(pred_idx),
            "true_label": class_names[true_idx],
            "pred_label": class_names[pred_idx],
            "confidence": float(confidence[idx]),
        })

    return misclassified


# Visualise misclassified images with GradCAM++ side-by-side

def plot_misclassified_gradcam(
    model,
    misclassified,
    last_conv_layer_name=None,
    max_display=12,
    save_path=None,
    figsize_per_row=(14, 3.5),
):
    samples = misclassified[:max_display]
    n = len(samples)
    if n == 0:
        print("No misclassified samples found — model is perfect!")
        return

    cols = 3   # original | grad-true | grad-pred
    fig_w, fig_h_row = figsize_per_row
    fig = plt.figure(figsize=(fig_w, fig_h_row * n), facecolor="#1a1a2e")

    col_titles = ["Original", "GradCAM++ (True class)", "GradCAM++ (Predicted class)"]

    for row, sample in enumerate(samples):
        img       = sample["img"]          # (H, W, 3) float32
        true_idx  = sample["true_idx"]
        pred_idx  = sample["pred_idx"]
        true_lbl  = sample["true_label"]
        pred_lbl  = sample["pred_label"]
        conf      = sample["confidence"]

        # GradCAM++ heatmaps
        heat_true = get_gradcam_pp(model, img, true_idx,  last_conv_layer_name)
        heat_pred = get_gradcam_pp(model, img, pred_idx,  last_conv_layer_name)

        overlay_true = overlay_heatmap(img, heat_true)
        overlay_pred = overlay_heatmap(img, heat_pred)

        panels = [
            (np.uint8(sample["img_display"] * 255), f"True: {true_lbl}", "#4ecca3"),
            (overlay_true,        f"GradCAM++ → {true_lbl}",  "#4ecca3"),
            (overlay_pred,        f"Pred: {pred_lbl}  ({conf:.1%})", "#e94560"),
        ]

        for col, (panel_img, subtitle, color) in enumerate(panels):
            ax = fig.add_subplot(n, cols, row * cols + col + 1)
            ax.imshow(panel_img)
            ax.set_title(subtitle, fontsize=9, color=color, fontweight="bold", pad=4)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

    # Reserve a fixed ~0.6in band at the top for the title + column headers,
    # and lay out the panel grid BELOW it first (so tight_layout can't reclaim
    # the band and collide the header text — the original bug).
    header_band = 0.6 / (fig_h_row * n)        # fraction of fig height
    top = 1.0 - header_band
    plt.tight_layout(rect=[0, 0, 1, top])

    # Column headers — lower row of the reserved band
    for col, title in enumerate(col_titles):
        fig.text(
            (col + 0.5) / cols, top + header_band * 0.30,
            title,
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            color="white",
        )

    # Figure title — upper row of the reserved band (above the headers)
    fig.suptitle(
        "Misclassified Samples with GradCAM++",
        fontsize=14, fontweight="bold",
        color="white", y=top + header_band * 0.75,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"✅ Saved → {save_path}")

    plt.show()

# Summary bar: how many per (true, pred) pair were misclassified

def plot_misclassification_summary(misclassified, class_names, save_path=None):
    from collections import Counter
    pairs = Counter(
        (s["true_label"], s["pred_label"]) for s in misclassified
    )
    labels = [f"{t}→{p}" for (t, p) in pairs]
    counts = list(pairs.values())

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    bars = ax.bar(labels, counts, color="#e94560", edgecolor="#4ecca3", linewidth=1.2)
    ax.bar_label(bars, padding=3, color="white", fontsize=10)
    ax.set_xlabel("True → Predicted", color="white", fontsize=11)
    ax.set_ylabel("Count", color="white", fontsize=11)
    ax.set_title("Misclassification Pairs", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4ecca3")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"✅ Saved → {save_path}")

    plt.show()

# Swin Modify for Gradcam++ 

def get_swin_target_layer(model):
    target = None
    for layer in model.layers:
        # The tfswin backbone is a single nested model
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                if isinstance(sublayer, tf.keras.layers.LayerNormalization):
                    target = sublayer.name
    return target

def get_gradcam_pp_swin(model, img_array, class_idx):
    # Find the GAP layer input shape to know the feature map size
    # Run one forward pass to get the intermediate feature map
    img_4d = tf.cast(img_array[np.newaxis, ...], tf.float32)

    # Build a feature extractor up to GAP using the functional graph
    gap_layer_name  = 'global_avg_pool'
    swin_layer_name = None

    # Find the Swin backbone layer name
    for layer in model.layers:
        if 'swin' in layer.name.lower() or 'tiny' in layer.name.lower():
            swin_layer_name = layer.name
            break

    if swin_layer_name is None:
        raise ValueError("Could not find Swin backbone layer in model.")

    # Run forward pass manually layer by layer to get feature map
    x = img_4d
    feat_map = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if layer.name == gap_layer_name:
            feat_map = x   # capture feature map BEFORE GAP
            break
        x = layer(x, training=False)

    if feat_map is None:
        raise ValueError("Could not capture feature map before GAP layer.")

    # Now use tf.Variable on the feature map for gradient computation
    feat_var = tf.Variable(feat_map)

    # Build the head (everything from GAP onward)
    found_gap = False
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            h = feat_var
            for layer in model.layers:
                if found_gap:
                    h = layer(h, training=False)
                if layer.name == gap_layer_name:
                    found_gap = True
                    h = layer(h, training=False)
            score = h[:, class_idx]
        grads1 = tape1.gradient(score, feat_var)
    grads2 = tape2.gradient(grads1, feat_var)

    feat_np = feat_map.numpy()[0]   # (7, 7, 768)
    g1      = grads1.numpy()[0]
    g2      = grads2.numpy()[0]

    g3  = g2 * g1 + 1e-8
    num = g2
    den = 2.0 * g2 + np.sum(feat_np * g3, axis=(0, 1), keepdims=True)
    den = np.where(np.abs(den) < 1e-8, 1e-8, den)

    alpha   = num / den
    weights = np.sum(alpha * np.maximum(g1, 0), axis=(0, 1))

    cam = np.sum(feat_np * weights, axis=-1)    # (7, 7)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.astype(np.float32)

