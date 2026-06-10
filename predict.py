import torch
import tifffile
import numpy as np
import matplotlib.pyplot as plt

from attention_unet import build_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(path):
    img = tifffile.imread(path)

    if img.ndim == 3:
        if img.shape[0] == 2:
            img = img[0]
        elif img.shape[-1] == 2:
            img = img[:, :, 0]

    img = img.astype(np.float32)

    img = (img - img.mean()) / (img.std() + 1e-8)

    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)

    return img.to(DEVICE)


def predict(model_path, image_path):

    model = build_model("pretrained", DEVICE)

    checkpoint = torch.load(
    model_path,
    map_location=DEVICE,
    weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    image = load_image(image_path)

    with torch.no_grad():
        pred = model(image)

    pred = pred.squeeze().cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8)

    return pred, mask


if __name__ == "__main__":

    pred, mask = predict(
        "checkpoints/best_model.pth",
        "data/testImage_ph3.tif"
    )

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(pred, cmap="gray")
    plt.title("Probability Map")

    plt.subplot(1,2,2)
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Mask")

    plt.tight_layout()

    plt.savefig("results/test_prediction.png")

    print("Prediction saved.")