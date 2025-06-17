import streamlit as st
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw
import io
import plotly.graph_objs as go

def lift_plateaus(y, sat=254, min_len=4, gain=1.2):
    y = y.astype(float).copy()
    n = len(y)
    i = 0
    while i < n:
        if y[i] >= sat:
            start = i
            while i < n and y[i] >= sat:
                i += 1
            end = i - 1
            if end - start + 1 >= min_len:
                left = y[start - 1] if start > 0 else y[end + 1]
                right = y[end + 1] if end + 1 < n else y[start - 1]
                peak = max(left, right) * gain
                mid = (start + end) // 2
                for j in range(start, end + 1):
                    if j <= mid:
                        y[j] = left + (peak - left) * (j - start + 1) / (mid - start + 1)
                    else:
                        y[j] = right + (peak - right) * (end - j + 1) / (end - mid + 1)
        else:
            i += 1
    return y

def extract_profile(img_bgr, roi, method="lum"):
    x, y, w, h = map(int, roi)
    strip = img_bgr[y:y + h, x:x + w]
    if strip.size == 0:
        raise ValueError("Empty ROI – try a bigger rectangle")
    def _smooth(v):
        v = v.astype(float)
        v -= v.min()
        return gaussian_filter1d(v, sigma=2)
    if method == "lum":
        lum = 0.299 * strip[:, :, 2] + 0.587 * strip[:, :, 1] + 0.114 * strip[:, :, 0]
        return _smooth(lum.mean(axis=0))
    if method == "rgb":
        return {
            "r": _smooth(strip[:, :, 2].mean(axis=0)),
            "g": _smooth(strip[:, :, 1].mean(axis=0)),
            "b": _smooth(strip[:, :, 0].mean(axis=0)),
        }
    ch = {"b": 0, "g": 1, "r": 2}[method.lower()]
    return _smooth(strip[:, :, ch].mean(axis=0))

def blend_channels(r, g, b, delta_ratio=0.10):
    r, g, b = map(np.asarray, (r, g, b))
    comp = np.empty_like(r, dtype=float)
    rgb = np.vstack([r, g, b])
    for i in range(rgb.shape[1]):
        vals = rgb[:, i]
        top, second = np.partition(vals, -2)[-2:]
        comp[i] = top if top - second > delta_ratio * top else (top + second) / 2
    comp -= comp.min()
    return gaussian_filter1d(comp, sigma=2)

def draw_crop_rectangle(img, left, top, right, bottom):
    """Draws a rectangle on the image to show the current crop."""
    draw = ImageDraw.Draw(img)
    draw.rectangle([left, top, right, bottom], outline='red', width=3)
    return img

st.set_page_config(layout="centered")
st.title("📸 Crop, Calibrate & Visualize Spectrum (RGB and Composite)")

# Initialize session state for saved crop and calibration
if "saved_crop" not in st.session_state:
    st.session_state["saved_crop"] = None
    st.session_state["saved_img"] = None
    st.session_state["calib_px1"] = 0
    st.session_state["calib_px2"] = 0  # Will be updated after crop is saved
    st.session_state["calib_wl1"] = 450.0
    st.session_state["calib_wl2"] = 625.0
    st.session_state["calib_coeff"] = None

option = st.radio("Choose input method:", ("Camera", "Upload Image"), horizontal=True)
if option == "Camera":
    uploaded_img = st.camera_input("Take a photo")
else:
    uploaded_img = st.file_uploader("Upload a spectrum image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_img:
    img = Image.open(io.BytesIO(uploaded_img.getvalue()))
    st.image(img, caption="Original Image", use_container_width=True)

    st.subheader("Crop Settings")
    col1, col2 = st.columns(2)
    with col1:
        left = st.slider("Left", 0, img.width-1, 0, key="slider_left")
        top = st.slider("Top", 0, img.height-1, 0, key="slider_top")
    with col2:
        right = st.slider("Right", left+1, img.width, img.width, key="slider_right")
        bottom = st.slider("Bottom", top+1, img.height, img.height, key="slider_bottom")

    # Draw crop rectangle on the image for real-time feedback
    img_with_rect = img.copy()
    img_with_rect = draw_crop_rectangle(img_with_rect, left, top, right, bottom)
    st.image(img_with_rect, caption="Crop Preview (rectangle updates in real time)", use_container_width=True)

    if st.button("💾 Save Crop"):
        if right > left and bottom > top:
            cropped = img.crop((left, top, right, bottom))
            st.session_state["saved_crop"] = (left, top, right, bottom)
            st.session_state["saved_img"] = cropped
            st.success("Crop saved! Plots will now use this crop.")
        else:
            st.warning("Right must be greater than Left and Bottom must be greater than Top for cropping.")

    # Use saved crop if available
    if st.session_state["saved_img"] is not None:
        cropped = st.session_state["saved_img"]
        st.image(cropped, caption="Saved Cropped Result", use_container_width=True)

        # --- Spectrum Extraction ---
        cropped_np = np.array(cropped)
        if cropped_np.ndim == 2:  # grayscale
            cropped_np = cv2.cvtColor(cropped_np, cv2.COLOR_GRAY2RGB)
        cropped_bgr = cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR)
        roi = (0, 0, cropped_bgr.shape[1], cropped_bgr.shape[0])
        prof = extract_profile(cropped_bgr, roi, method="rgb")
        r = lift_plateaus(prof["r"])
        g = lift_plateaus(prof["g"])
        b = lift_plateaus(prof["b"])
        comp = blend_channels(r, g, b)
        x_axis = np.arange(len(r))
        st.session_state["calib_px2"] = len(comp) - 1  # Update after crop is saved

        # --- Calibration with Interactive Sliders ---
        st.subheader("Wavelength Calibration (Interactive)")
        st.markdown("Move the sliders to align the vertical lines with known spectral features. Enter the corresponding wavelengths.")
        st.markdown("**Note:** while caliberating, the peak of blue is at 450 nm and the peak of red is at 625 nm. You can use these values to calibrate the spectrum.")

        px1 = st.slider("Pixel λ₁", 0, len(comp)-1, st.session_state["calib_px1"], key="slider_px1")
        px2 = st.slider("Pixel λ₂", 0, len(comp)-1, st.session_state["calib_px2"], key="slider_px2")
        wl1 = st.number_input("Wavelength at Pixel λ₁ (nm)", min_value=1.0, value=st.session_state["calib_wl1"], step=1.0, key="num_wl1")
        wl2 = st.number_input("Wavelength at Pixel λ₂ (nm)", min_value=1.0, value=st.session_state["calib_wl2"], step=1.0, key="num_wl2")

        if st.button("Save Calibration"):
            if px1 != px2:
                st.session_state["calib_px1"] = px1
                st.session_state["calib_px2"] = px2
                st.session_state["calib_wl1"] = wl1
                st.session_state["calib_wl2"] = wl2
                st.session_state["calib_coeff"] = np.polyfit([px1, px2], [wl1, wl2], 1)
                st.success("Calibration saved and will be reused for subsequent spectra.")
            else:
                st.warning("Select two different pixels for calibration.")

        coeff = st.session_state.get("calib_coeff")
        if coeff is not None:
            wavelengths = np.polyval(coeff, x_axis)

            # RGB plot (wavelength vs intensity, interactive)
            fig_rgbly = go.Figure()
            fig_rgbly.add_trace(go.Scatter(x=wavelengths, y=r, mode='lines', name='Red', line=dict(color='red')))
            fig_rgbly.add_trace(go.Scatter(x=wavelengths, y=g, mode='lines', name='Green', line=dict(color='green')))
            fig_rgbly.add_trace(go.Scatter(x=wavelengths, y=b, mode='lines', name='Blue', line=dict(color='blue')))
            # Add vertical lines for current calibration points (real-time)
            wl_px1 = wavelengths[px1] if px1 < len(wavelengths) else wavelengths[-1]
            wl_px2 = wavelengths[px2] if px2 < len(wavelengths) else wavelengths[-1]
            fig_rgbly.add_vline(x=wl_px1, line_color="magenta", line_dash="dash", annotation_text="λ₁")
            fig_rgbly.add_vline(x=wl_px2, line_color="cyan", line_dash="dash", annotation_text="λ₂")
            fig_rgbly.update_layout(
                title="Calibrated Spectrum (Wavelength vs. Intensity, RGB)",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Intensity",
                hovermode="x unified"
            )
            st.plotly_chart(fig_rgbly, use_container_width=True)

            # Composite plot (wavelength vs intensity, interactive)
            fig_comply = go.Figure()
            fig_comply.add_trace(go.Scatter(x=wavelengths, y=comp, mode='lines', name='Composite', line=dict(color='#444')))
            # Add vertical lines for current calibration points (real-time)
            fig_comply.add_vline(x=wl_px1, line_color="magenta", line_dash="dash", annotation_text="λ₁")
            fig_comply.add_vline(x=wl_px2, line_color="cyan", line_dash="dash", annotation_text="λ₂")
            fig_comply.update_layout(
                title="Calibrated Spectrum (Wavelength vs. Intensity, Composite)",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Intensity",
                hovermode="x unified"
            )
            st.plotly_chart(fig_comply, use_container_width=True)

            # Download options
            df = pd.DataFrame({
                "pixel": x_axis,
                "wavelength_nm": wavelengths,
                "intensity_red": r,
                "intensity_green": g,
                "intensity_blue": b,
                "intensity_composite": comp
            })
            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button("Download CSV", csv_buffer.getvalue(), "spectrum.csv", "text/csv")
        else:
            st.info("Please calibrate and save to enable wavelength plots.")
    else:
        st.info("Please save a crop to proceed.")
else:
    st.info("Please upload or capture an image to proceed.")
