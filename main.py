import matplotlib
matplotlib.use('TkAgg')  #  Fix for image display

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import heapq
import os
import matplotlib.pyplot as plt
import pickle

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad


# ==================================================
# 1️⃣ Load Image
# ==================================================
image_path = input("Enter image path (PNG/JPG/BMP): ").strip()

if not os.path.isfile(image_path):
    print("Invalid file path")
    exit()

img = Image.open(image_path).convert("RGB")

img.save("temp.bmp")
original_size = os.path.getsize("temp.bmp")

img_array = np.array(img)

img_tensor = torch.tensor(img_array/255.0,dtype=torch.float32).permute(2,0,1).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_tensor = img_tensor.to(device)


# ==================================================
# 2️⃣ Neural Network Predictor
# ==================================================
class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,3,3,padding=1)
        )

    def forward(self,x):
        return self.net(x)


model = Predictor().to(device)

#  Load trained model (instead of training again)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

print(" Model loaded instantly!")


# ==================================================
# 3️⃣ Residual Calculation
# ==================================================
with torch.no_grad():
    prediction = model(img_tensor)

prediction = (prediction.squeeze().cpu().numpy()*255).round().astype(int)
prediction = np.clip(prediction,0,255)

original = (img_tensor.squeeze().cpu().numpy()*255).round().astype(int)

residual = original - prediction


# ==================================================
# 4️⃣ Huffman Coding
# ==================================================
def build_huffman(data):

    freq = {}
    for v in data.flatten():
        freq[v] = freq.get(v,0)+1

    heap = [[weight,[symbol,""]] for symbol,weight in freq.items()]
    heapq.heapify(heap)

    while len(heap)>1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)

        for pair in lo[1:]:
            pair[1] = '0'+pair[1]

        for pair in hi[1:]:
            pair[1] = '1'+pair[1]

        heapq.heappush(heap,[lo[0]+hi[0]]+lo[1:]+hi[1:])

    return dict(heapq.heappop(heap)[1:])


codes = build_huffman(residual)

#  Save Huffman codes
with open("codes.pkl","wb") as f:
    pickle.dump(codes,f)


# ==================================================
# 5️⃣ Bitstream to Bytes (SAFE)
# ==================================================
bitstream = ''.join(codes[v] for v in residual.flatten())
bit_length = len(bitstream)

byte_array = bytearray()
for i in range(0, len(bitstream), 8):
    byte = bitstream[i:i+8]
    byte = byte.ljust(8, '0')
    byte_array.append(int(byte, 2))

byte_array = bytes(byte_array)


# ==================================================
# 6️⃣ AES Encryption
# ==================================================
key = get_random_bytes(16)

#  Save key
with open("key.bin","wb") as f:
    f.write(key)

cipher = AES.new(key,AES.MODE_CBC)

ciphertext = cipher.encrypt(pad(byte_array,AES.block_size))

with open("compressed_secure.bin","wb") as f:
    f.write(cipher.iv)
    f.write(ciphertext)

bin_size = os.path.getsize("compressed_secure.bin")
codes_size = os.path.getsize("codes.pkl")
key_size = os.path.getsize("key.bin")

compressed_size = bin_size + codes_size + key_size

print("Secure compressed file saved")


# ==================================================
# 7️⃣ AES Decryption
# ==================================================
with open("key.bin","rb") as f:
    key = f.read()

with open("compressed_secure.bin","rb") as f:
    iv = f.read(16)
    ciphertext = f.read()

cipher = AES.new(key,AES.MODE_CBC,iv)

decrypted = unpad(cipher.decrypt(ciphertext),AES.block_size)


# ==================================================
# 8️⃣ Bytes to Bitstream
# ==================================================
bitstream = ''.join(f"{byte:08b}" for byte in decrypted)
bitstream = bitstream[:bit_length]


# ==================================================
# 9️⃣ Huffman Decode
# ==================================================
with open("codes.pkl","rb") as f:
    codes = pickle.load(f)

reverse_codes = {v:k for k,v in codes.items()}

decoded = []
buffer=""

for bit in bitstream:
    buffer+=bit
    if buffer in reverse_codes:
        decoded.append(reverse_codes[buffer])
        buffer=""

decoded = np.array(decoded).reshape(residual.shape)


# ==================================================
# 🔟 Reconstruct Image
# ==================================================
reconstructed = prediction + decoded

reconstructed = np.clip(reconstructed,0,255).astype(np.uint8)
reconstructed = np.transpose(reconstructed,(1,2,0))

reconstructed_img = Image.fromarray(reconstructed)
reconstructed_img.save("reconstructed_compressed.jpg", quality=50, optimize=True)
final_size = os.path.getsize("reconstructed_compressed.jpg")

print(f"\nOriginal Size : {original_size/1024:.2f} KB")
print(f"Compressed Size : {compressed_size/1024:.2f} KB")

compression_ratio = original_size / compressed_size
print(f"Compression Ratio : {compression_ratio:.2f}x")

# ==================================================
# PIXEL COMPARISON (LOSSLESS CHECK)
# ==================================================

original_img_uint8 = np.transpose(original, (1,2,0)).astype(np.uint8)

pixel_equal = np.array_equal(original_img_uint8, reconstructed)

print("\nPIXEL COMPARISON RESULT:")
print("Are all pixels exactly same? ->", pixel_equal)

diff_count = np.sum(original_img_uint8 != reconstructed)
print("Number of different pixel values:", diff_count)

# Sample pixel values
print("\nSample Pixel Values (Original vs Reconstructed):")

flat_orig = original_img_uint8.reshape(-1,3)
flat_recon = reconstructed.reshape(-1,3)

num_pixels = int(input("How many pixels to display? "))
for i in range(num_pixels):
#for i in range(10):
    print(f"Pixel {i}: Original={flat_orig[i]}  Reconstructed={flat_recon[i]}")

# Difference image
difference = np.abs(original_img_uint8.astype(int) - reconstructed.astype(int)).astype(np.uint8)


# ==================================================
# 11️⃣ Results
# ==================================================



# ==================================================
# 12️⃣ DISPLAY ALL IMAGES TOGETHER
# ==================================================
plt.figure(figsize=(15,5))

# Original
plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

# Reconstructed
plt.subplot(1,3,2)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_img)
plt.axis("off")

# Difference
difference = np.abs(original_img_uint8.astype(int) - reconstructed.astype(int)).astype(np.uint8)

plt.subplot(1,3,3)
plt.title("Difference (Black = Lossless)")
plt.imshow(difference)
plt.axis("off")

plt.tight_layout()
plt.show()
