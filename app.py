# --- Imports ---
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from brisque import BRISQUE
import pyiqa
import torch
from PIL import Image
from io import BytesIO
import zipfile

# --- SCUNet Imports (pastikan sudah tersedia) ---
from models.network_scunet import SCUNet as net
from utils import utils_image as util

# --- Setup Device ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
iqa_metric = pyiqa.create_metric('niqe').to(device)

# --- Load SCUNet Model ---
scunet_model = net(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
scunet_model.load_state_dict(torch.load('color_images-50.pth', map_location=device), strict=True)
scunet_model = scunet_model.to(device)
scunet_model.eval()

# --- Load SCUNet Model (real image denoising) ---
scunet_model_real = net(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
scunet_model_real.load_state_dict(torch.load('real_image_denoising.pth', map_location=device), strict=True)
scunet_model_real = scunet_model_real.to(device)
scunet_model_real.eval()

# --- Fungsi Enhancement ---
def SSR(img, sigma):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    if img.ndim != blur.ndim:
        blur = np.expand_dims(blur, axis=-1)
    retinex = np.log1p(img) - np.log1p(blur)
    return retinex

def MSR(img, sigma_list):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += SSR(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex

def color_restoration(img, alpha=125, beta=46):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_rest = beta * (np.log1p(alpha * img) - np.log1p(img_sum))
    return color_rest

def MSRCR(img, sigma_list, G=5, b=25, alpha=125, beta=46, low_clip=0.01, high_clip=0.99):
    img = img.astype(np.float32) + 1.0
    retinex = MSR(img, sigma_list)
    color_rest = color_restoration(img, alpha, beta)
    msrcr = G * (retinex * color_rest + b)
    for i in range(msrcr.shape[2]):
        channel = msrcr[:, :, i]
        channel = np.clip(channel, np.percentile(channel, low_clip * 100), np.percentile(channel, high_clip * 100))
        channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel)) * 255
        msrcr[:, :, i] = channel
    msrcr = np.clip(msrcr, 0, 255).astype(np.uint8)
    return msrcr

def MSRCP(img, sigma_list, low_clip=0.01, high_clip=0.99):
    img = img.astype(np.float32) + 1.0
    intensity = np.mean(img, axis=2)
    retinex = MSR(np.expand_dims(intensity, axis=2), sigma_list)
    retinex = np.squeeze(retinex)
    retinex = np.clip(retinex, np.percentile(retinex, low_clip * 100), np.percentile(retinex, high_clip * 100))
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255
    img_cp = np.zeros_like(img)
    for i in range(3):
        img_cp[:, :, i] = retinex * (img[:, :, i] / intensity)
    img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
    return img_cp

# --- Fungsi Penilaian ---
def calculate_brisque_from_array(image_array):
    brisque_model = BRISQUE()
    return brisque_model.score(image_array)

def calculate_niqe_from_array(image_array):
    img_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)
    niqe_score = iqa_metric(img_tensor)
    return niqe_score.item()

# --- Fungsi Denoising dengan SCUNet ---
def run_scunet(img, model):
    img_input = util.uint2tensor4(img)
    img_input = img_input.to(device)
    with torch.no_grad():
        img_output = model(img_input)
    img_output = util.tensor2uint(img_output)
    return img_output



# --- Streamlit UI ---
st.title("Retinex Image Enhancement + Optional Denoising with SCUNet")

uploaded_file = st.file_uploader("Upload a JPG/PNG Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file).convert('RGB')
    img = np.array(img)

    sigma_list = [15, 80, 250]

    # Process Retinex
    img_ssr = SSR(img.astype(np.float32) + 1.0, sigma=80)
    img_ssr = exposure.rescale_intensity(img_ssr, out_range=(0, 255)).astype(np.uint8)

    img_msr = MSR(img.astype(np.float32) + 1.0, sigma_list)
    img_msr = exposure.rescale_intensity(img_msr, out_range=(0, 255)).astype(np.uint8)

    img_msrcr = MSRCR(img, sigma_list)
    img_msrcp = MSRCP(img, sigma_list)

    all_images = {
        'Original': img,
        'SSR': img_ssr,
        'MSR': img_msr,
        'MSRCR': img_msrcr,
        'MSRCP': img_msrcp
    }

    # Checkbox untuk Denoising
    denoising_method = st.selectbox(
        'Select Denoising Method',
        ('None', 'High Color Denoising', 'Real Image Denoising')
    )

    results = {}

    # Jika Denoising dipilih
    if denoising_method != 'None':
        enhanced_images = {}
        for key, image in all_images.items():
            if denoising_method == 'High Color Denoising':
                denoised = run_scunet(image, scunet_model)
            elif denoising_method == 'Real Image Denoising':
                denoised = run_scunet(image, scunet_model_real)
            enhanced_images[key + f' + {denoising_method}'] = denoised
        all_images = enhanced_images


    # Hitung skor dan tampilkan
    st.subheader("Processed Images and Quality Scores")
    cols = st.columns(3)

    for idx, (key, image) in enumerate(all_images.items()):
        with cols[idx % 3]:
            brisque = calculate_brisque_from_array(image)
            niqe = calculate_niqe_from_array(image)
            st.image(image, caption=f"{key}\nBRISQUE: {brisque:.2f}, NIQE: {niqe:.2f}", use_container_width=True)
             # Siapkan gambar untuk diunduh
            img_pil = Image.fromarray(image)
            buf = BytesIO()
            img_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            # Tempatkan tombol di tengah
            button_col = st.columns([1, 2, 1])[1]  # Kolom tengah dari total 3 kolom
            with button_col:
                st.download_button(
                    label=f"Download",
                    data=byte_im,
                    file_name=f"{key.replace(' ', '_').lower()}.png",
                    mime="image/png",
                    use_container_width=True
                )
    
    # --- Buat ZIP dari semua gambar ---
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        for key, image in all_images.items():
            img_pil = Image.fromarray(image)
            img_bytes = BytesIO()
            img_pil.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            zip_file.writestr(f"{key.replace(' ', '_').lower()}.png", img_bytes.read())

    zip_buffer.seek(0)

    # Tampilkan tombol download ZIP
    st.markdown("---")
    st.markdown("### 📦 Download Semua Hasil")
    st.download_button(
        label="⬇ Download All as ZIP",
        data=zip_buffer,
        file_name="enhanced_images.zip",
        mime="application/zip",
        use_container_width=True
    )