import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import contact

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="AFM Force Curve Viewer", layout="wide")

st.title("🔬 AFM Force Curve Contact Point Viewer")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Settings")

# Selectable file path
FC_path = st.sidebar.text_input("Enter Force Curve File Path:", value="/Volumes/ExtremeSSD/nojima/251015/1442_0/ForceCurve.tdms")

# Contact preprocessor parameters
st.sidebar.subheader("Contact Detection Parameters")
p            = st.sidebar.number_input("p", value=1.5)
sigma_thresh = st.sidebar.number_input("sigma_thresh", value=1.5)
fit_span     = st.sidebar.number_input("fit_span", value=0.5)


# -----------------------------
# Data Loading and Preprocessing
# -----------------------------
if FC_path and FC_path.strip() != "":
    try:
        prepro = contact.AFMForceCurvePreprocessor(
            contact_params=dict(p=p, sigma_thresh=sigma_thresh, fit_span=fit_span)
        )

        # Run preprocessing once
        st.write("📂 Loading data...")
        test = prepro.preprocess_curve(FC_path, index=0)
        slen = test["lenData"]

        # Select index interactively
        index = st.sidebar.slider("Curve Index", 0, slen - 1, 1000)

        # Process selected index
        test = prepro.preprocess_curve(FC_path, index=index)

        # -----------------------------
        # Fast Plot
        # -----------------------------
        fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
        ax.plot(test["z_left"], test["d_left"] / 1000, ".", label="F(z)")
        ax.plot(test["z_left"], test["d_smooth"] / 1000, "-k")
        ax.plot(test["Delta"], test["Force"] / 1000, ".", label=r"F($\delta$): $\delta$ = z - Dk")
        ax.plot(test["Delta"], test["ForceS"] / 1000, "-k")
        ax.axvline(test["cp"], color="k", ls="--", label="contact")
        ax.set_xlabel(r"z|$\delta$ [nm]")
        ax.set_ylabel(r"F [$\mu$N]")
        ax.set_title("Force Curve - Contact point - depth $\delta$", loc="left")
        ax.legend(frameon=False)

        st.pyplot(fig, clear_figure=True)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("Please enter a valid file path to begin.")
