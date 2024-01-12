import idx2numpy
import matplotlib.pyplot as plt

train_images = idx2numpy.convert_from_file(
    "data/FashionMNIST/raw/t10k-images-idx3-ubyte"
)

print(f"Shape of train images: {train_images.shape}")

for idx in range(10):
    plt.imshow(train_images[idx], cmap="gray")
    plt.title("test label")
    plt.show()
