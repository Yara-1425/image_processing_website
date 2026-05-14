import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



# Gaussian
def add_gaussian_noise(image, sigma):
    row, col, ch = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Salt & Pepper
def add_salt_and_pepper_noise(image, amount):
    noisy = image.copy()
    # Salt mode
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords[:2])] = 255
    # Pepper mode
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords[:2])] = 0
    return noisy

#Page Settings
st.set_page_config(page_title="image processing lab", layout="wide", initial_sidebar_state="expanded")

# Caching
@st.cache_data
def load_base_image(uploaded_file):
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    return img_array

#Sidebar - Upload the image only once
with st.sidebar:
    st.title("Control Panel")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Image to Start:", type=['jpg', 'png', 'jpeg'])

    if uploaded_file:
        st.success("Image Uploaded Successfully!")
    else:
        st.info("Waiting for image upload...")

#Main title
st.title("Image Processing Lab")
st.markdown("Developed by **Alanoud, Shahad, Lamis, Nada, Yara**")

# Create the 5 tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Essentials",
    "Spatial Enhancement",
    "Frequency Domain",
    "Edges & Segmentation",
    "Morphology & Analysis"
])
if uploaded_file:
    original_img = load_base_image(uploaded_file)

    #Tab 1: Essentials
    with tab1:
        st.header("Image Fundamentals")
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(original_img, caption="Original View", use_container_width=True)
        with col_b:
            st.subheader("Image Metadata")
            height, width = original_img.shape[:2]

            channels = original_img.shape[2]

            st.write(f"Width : {width} pixels")

            st.write(f"Height : {height} pixels")

            st.write(f"Channels : {channels}")

            st.write(f"Data Type : {original_img.dtype}")

            st.write(f"Image Shape : {original_img.shape}")

        st.divider()

        # =====================================
        # Color Space Conversion
        # =====================================

        st.subheader("Color Space Conversion")

        # اختيار نوع Color Space
        color_option = st.selectbox(
            "Choose Color Space",
            ["RGB", "Grayscale", "HSV"]
        )

        # =====================================
        # معلومات عن Color Space
        # =====================================

        if color_option == "RGB":

            st.info("""
            RGB stands for Red, Green, and Blue.

            It is the standard color space used for displaying images.

            Each color is created by combining these three channels.
            """)

            converted_img = original_img

        elif color_option == "Grayscale":

            st.info("""
            Grayscale converts the image into shades of gray.

            It reduces the image from 3 channels to 1 channel.

            Commonly used in:
            - Edge Detection
            - Thresholding
            - Computer Vision tasks
            """)

            converted_img = cv2.cvtColor(
                original_img,
                cv2.COLOR_RGB2GRAY
            )

        elif color_option == "HSV":

            st.info("""
            HSV stands for Hue, Saturation, and Value.

            It separates color information from brightness.

            Commonly used in:
            - Color Tracking
            - Object Detection
            - Image Segmentation
            """)

            converted_img = cv2.cvtColor(
                original_img,
                cv2.COLOR_RGB2HSV
            )

        # عرض الصورة الناتجة
        st.image(
            converted_img,
            caption=f"{color_option} Image",
            use_container_width=True
        )

        st.divider()

        # =====================================
        # Histogram
        # =====================================

        st.subheader("Histogram")

        fig, ax = plt.subplots(figsize=(8,4))

        # إذا كانت الصورة Grayscale
        if len(converted_img.shape) == 2:

            ax.hist(
                converted_img.ravel(),
                bins=256,
                range=(0,256)
            )

            ax.set_title("Grayscale Histogram")

        # إذا كانت الصورة ملونة
        else:

            colors = ('r', 'g', 'b')

            for i, color in enumerate(colors):

                hist = cv2.calcHist(
                    [converted_img],
                    [i],
                    None,
                    [256],
                    [0,256]
                )

                ax.plot(hist, color=color)

            ax.set_title("Color Histogram")

        ax.set_xlabel("Pixel Value")

        ax.set_ylabel("Frequency")

        st.pyplot(fig)



    # Tab 2: Spatial Enhancement
    with tab2:
        st.header("Spatial Domain Enhancement")
        st.markdown("Direct pixel manipulation to improve visual quality or reduce noise.")

        # نستخدم أعمدة لتنظيم الواجهة: عمود للتحكم وعمود للنتائج
        col_ctrl, col_res = st.columns([1, 2])

        with col_ctrl:
            st.subheader(" Settings")

            st.markdown("### Add Noise")
            noise_type = st.selectbox("Select Noise Type", ["None", "Gaussian", "Salt and Pepper"])

            temp_img = original_img.copy()

            if noise_type == "Gaussian":
                sigma = st.slider("Noise Intensity (Sigma)", 0, 100, 25)
                temp_img = add_gaussian_noise(temp_img, sigma)
            elif noise_type == "Salt and Pepper":
                 amount = st.slider("Noise Density", 0.0, 0.5, 0.05)
                 temp_img = add_salt_and_pepper_noise(temp_img, amount)

            st.markdown("---")

            #Contrast & Brightness
            st.markdown("### Brightness & Contrast")
            alpha = st.slider("Contrast (Gain)", 1.0, 3.0, 1.0, 0.1)
            beta = st.slider("Brightness (Bias)", 0, 100, 0)
            st.caption("_Used to improve the visual appearance of images by expanding the range of intensity levels._")

            st.markdown("---")

            #Smoothing Filters
            st.markdown("### Filters (Denoising)")
            filter_type = st.selectbox("Select Filter",
                ["None", "Mean (Blur)", "Median", "Gaussian Blur", "Max Filter", "Min Filter"])
            kernel_size = st.select_slider("Kernel Size", options=[3, 5, 7, 9], value=3)


            if filter_type == "Mean (Blur)":
                st.caption("**Best for:** Reducing Gaussian noise. It averages pixels but blurs edges.")
            elif filter_type == "Median":
                st.caption("**Best for:** 'Salt & Pepper' noise. It preserves edges while removing spikes.")
            elif filter_type == "Gaussian Blur":
                st.caption("**Best for:** Natural smoothing and reducing random electronic noise.")
            elif filter_type == "Max Filter":
                st.caption("**Best for:** Removing 'Pepper' noise (dark spots) by expanding bright areas.")
            elif filter_type == "Min Filter":
                st.caption("**Best for:** Removing 'Salt' noise (bright spots) by expanding dark areas.")

            st.markdown("---")

            # Sharpening
            st.markdown("### Edge Sharpening")
            apply_sharpen = st.checkbox("Apply Laplacian Sharpening")

        with col_res:
            st.subheader("Result:")

            # تنفيذ العمليات بالتسلسل:
            # أ- تطبيق السطوع والتباين على الصورة (التي قد تحتوي على نويز)
            processed_img = cv2.convertScaleAbs(temp_img, alpha=alpha, beta=beta)

            # filter
            if filter_type == "Mean (Blur)":
                processed_img = cv2.blur(processed_img, (kernel_size, kernel_size))
            elif filter_type == "Median":
                processed_img = cv2.medianBlur(processed_img, kernel_size)
            elif filter_type == "Gaussian Blur":
                processed_img = cv2.GaussianBlur(processed_img, (kernel_size, kernel_size), 0)
            elif filter_type == "Max Filter":
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                processed_img = cv2.dilate(processed_img, kernel)
            elif filter_type == "Min Filter":
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                processed_img = cv2.erode(processed_img, kernel)

           #shrpness
            if apply_sharpen:
                kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                processed_img = cv2.filter2D(processed_img, -1, kernel_sharp)

            # عرض النتيجة النهائية
            st.image(processed_img, caption="Final Processed Image", use_container_width=True)


    #Tab 3: Frequency Domain
    with tab3:
        st.header("Fourier Transform & Frequency Filters")

        st.markdown("""
        Explore the image in the frequency domain using Fourier Transform.
        Frequency filters help us either smooth the image or enhance edges.
        """)

    # ============================================
    # Convert to Gray
    # ============================================
        gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

        st.subheader("1. Original Grayscale Image")
        st.image(gray, use_container_width=True)

        st.caption("""
    The image is converted to grayscale because frequency analysis works on intensity values only.
    """)

    # ============================================
    # FFT
    # ============================================
        st.subheader("2. Fast Fourier Transform (FFT Spectrum)")

    # Fourier Transform
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

    # Spectrum
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        magnitude_spectrum = cv2.normalize(
            magnitude_spectrum,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        magnitude_spectrum = magnitude_spectrum.astype(np.uint8)

        col1, col2 = st.columns(2)

        with col1:
            st.image(gray, caption="Input Image", use_container_width=True)

        with col2:
            st.image(
              magnitude_spectrum,
              caption="Frequency Spectrum",
              clamp=True,
              use_container_width=True
            )

        st.caption("""
    FFT transforms the image from spatial domain to frequency domain.
    Bright areas in the spectrum represent strong frequencies inside the image.
    """)

        st.markdown("---")

    # ============================================
    # Filter Selection
    # ============================================
        st.subheader("3. Frequency Filters")

        filter_type = st.selectbox(
            "Choose Filter Type",
        [
            "Ideal Low Pass",
            "Ideal High Pass",
            "Butterworth Low Pass",
            "Butterworth High Pass"
        ]
    )

        cutoff = st.slider(
            "Cutoff Frequency",
            min_value=5,
            max_value=100,
            value=30
    )

        order = st.slider(
            "Butterworth Order",
            min_value=1,
            max_value=10,
            value=2
    )

        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

    # Create meshgrid
        u = np.arange(rows)
        v = np.arange(cols)

        U, V = np.meshgrid(u, v, indexing='ij')

        D = np.sqrt((U - crow) ** 2 + (V - ccol) ** 2)

    # ============================================
    # Build Filter Mask
    # ============================================
        if filter_type == "Ideal Low Pass":
           mask = np.zeros((rows, cols), np.uint8)
           mask[D <= cutoff] = 1

           description = """
            Low Pass filters keep low frequencies and remove high frequencies.
            This produces a smoothing/blurring effect.
        """

        elif filter_type == "Ideal High Pass":
            mask = np.ones((rows, cols), np.uint8)
            mask[D <= cutoff] = 0

            description = """
            High Pass filters remove low frequencies and preserve edges/details.
            This enhances boundaries and sharp transitions.
            """

        elif filter_type == "Butterworth Low Pass":
            mask = 1 / (1 + (D / cutoff) ** (2 * order))

            description = """
            Butterworth Low Pass provides smoother transitions than Ideal filters.
            It reduces ringing artifacts while smoothing the image.
            """

        elif filter_type == "Butterworth High Pass":
            mask = 1 - (1 / (1 + (D / cutoff) ** (2 * order)))

            description = """
            Butterworth High Pass enhances edges gradually and avoids sharp artifacts.
            """

    # ============================================
    # Apply Filter
    # ============================================
        filtered_shift = fshift * mask

    # Inverse FFT
        f_ishift = np.fft.ifftshift(filtered_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

# Normalize for display
        img_back = cv2.normalize(
            img_back,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        img_back = img_back.astype(np.uint8)

    # ============================================
    # Display Results
    # ============================================
        st.subheader("4. Filter Results")

        colA, colB, colC = st.columns(3)

        with colA:
            st.image(gray, caption="Original", use_container_width=True)

        with colB:
            st.image(mask*255, caption="Filter Mask", clamp=True, use_container_width=True)

        with colC:
            st.image(
              img_back,
              caption="Filtered Image",
              clamp=True,
              use_container_width=True
           )

        st.caption(description)

    # ============================================
    # Extra Visualization
    # ============================================
        st.subheader("5. Filtered Spectrum")

        filtered_spectrum = 20 * np.log(np.abs(filtered_shift) + 1)

# Normalize spectrum
        filtered_spectrum = cv2.normalize(
            filtered_spectrum,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        filtered_spectrum = filtered_spectrum.astype(np.uint8)

        st.image(
            filtered_spectrum,
            caption="Spectrum After Applying Filter",
            clamp=True,
            use_container_width=True
        )

        st.caption("""
        This visualization shows which frequencies were preserved or removed after filtering.
        """)




    #Tab 4: Edges & Segmentation
    with tab4:
        st.header("Boundary Detection & Image Partitioning")
        st.markdown("""
        This section helps students understand how object boundaries are detected
        and how images can be separated into foreground and background.
        """)
        st.info("Implementation of Canny, Sobel, and Otsu Thresholding.")

        gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

        # تعريف المصفوفات (Kernels)
        prewitt_x_kernel = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ])

        prewitt_y_kernel = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ])

        st.subheader("Step 1: Grayscale Image")
        st.markdown("""
        **Why we use it:**
        Most edge detection and thresholding algorithms work better on grayscale images
        because they focus on intensity values instead of color information.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(gray_img, caption="Grayscale Image", use_container_width=True, clamp=True)

        st.markdown("---")

        # =========================
        # Edge Detection Section
        # =========================
        st.subheader("Step 2: Edge Detection")
        st.markdown("""
        **Why we use it:**
        Edge detection is used to find object boundaries in an image.
        """)

        edge_method = st.selectbox(
            "Choose Edge Detection Method:",
            ["Sobel", "Canny", "Prewitt"]
        )

        if edge_method == "Sobel":
            st.markdown("**Sobel Edge Detection:**")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                ksize = st.selectbox("Kernel Size", [3, 5, 7], index=0)
            with col_s2:
                scale = st.slider("Scale", 1, 5, 1)
            with col_s3:
                delta = st.slider("Delta", 0, 100, 0)

            sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta)
            sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta)
            sobel_x = cv2.convertScaleAbs(sobel_x)
            sobel_y = cv2.convertScaleAbs(sobel_y)
            sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                st.image(sobel_x, caption="Sobel X", use_container_width=True)
            with col_e2:
                st.image(sobel_y, caption="Sobel Y", use_container_width=True)
            with col_e3:
                st.image(sobel_combined, caption="Combined Sobel", use_container_width=True)

        elif edge_method == "Canny":
            st.markdown("**Canny Edge Detection:**")
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                threshold1 = st.slider("Lower Threshold", 0, 255, 100)
            with col_c2:
                threshold2 = st.slider("Upper Threshold", 0, 255, 200)
            with col_c3:
                blur_value = st.selectbox("Gaussian Blur Kernel", [3, 5, 7], index=0)

            blurred_img = cv2.GaussianBlur(gray_img, (blur_value, blur_value), 0)
            canny_edges = cv2.Canny(blurred_img, threshold1, threshold2)

            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.image(blurred_img, caption="Blurred", use_container_width=True)
            with col_e2:
                st.image(canny_edges, caption="Canny Result", use_container_width=True)

        elif edge_method == "Prewitt":
            st.markdown("**Prewitt Edge Detection:**")
            prewitt_x = cv2.filter2D(gray_img, -1, prewitt_x_kernel)
            prewitt_y = cv2.filter2D(gray_img, -1, prewitt_y_kernel)
            prewitt_combined = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.image(prewitt_x, caption="Prewitt X", use_container_width=True)
            with col_p2:
                st.image(prewitt_y, caption="Prewitt Y", use_container_width=True)
            with col_p3:
                st.image(prewitt_combined, caption="Combined Prewitt", use_container_width=True)

        st.markdown("---")

    # =========================
    # Thresholding Section
    # =========================
        st.subheader("Step 3: Thresholding")
        threshold_method = st.radio("Choose Thresholding Method:", ["Manual Threshold", "Otsu Threshold"], horizontal=True)

        if threshold_method == "Manual Threshold":
            threshold_value = st.slider("Threshold Value", 0, 255, 127)
            threshold_type = st.selectbox("Threshold Type", ["Binary", "Binary Inverted"])

            t_type = cv2.THRESH_BINARY if threshold_type == "Binary" else cv2.THRESH_BINARY_INV
            _, binary_img = cv2.threshold(gray_img, threshold_value, 255, t_type)

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.image(gray_img, caption="Grayscale Image", use_container_width=True)
            with col_t2:
                st.image(binary_img, caption=f"Manual Result ({threshold_value})", use_container_width=True)

        else:
            blur_for_otsu = st.checkbox("Apply Gaussian Blur Before Otsu", value=True)
            otsu_input = cv2.GaussianBlur(gray_img, (5, 5), 0) if blur_for_otsu else gray_img
            otsu_value, otsu_binary = cv2.threshold(otsu_input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            st.success(f"Otsu selected threshold value: {otsu_value:.2f}")
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                st.image(otsu_input, caption="Input for Otsu", use_container_width=True)
            with col_o2:
                st.image(otsu_binary, caption="Otsu Result", use_container_width=True)

        st.markdown("---")

    # =========================
    # Comparison Section
    # =========================
        st.subheader("Step 4: Quick Comparison")

        sobel_x_comp = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y_comp = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_comp = cv2.addWeighted(cv2.convertScaleAbs(sobel_x_comp), 0.5, cv2.convertScaleAbs(sobel_y_comp), 0.5, 0)
        canny_comp = cv2.Canny(gray_img, 100, 200)
        prew_comp = cv2.addWeighted(cv2.filter2D(gray_img, -1, prewitt_x_kernel), 0.5, cv2.filter2D(gray_img, -1, prewitt_y_kernel), 0.5, 0)
        _, otsu_comp = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        comp1, comp2, comp3, comp4 = st.columns(4)
        with comp1: st.image(sobel_comp, caption="Sobel", use_container_width=True)
        with comp2: st.image(canny_comp, caption="Canny", use_container_width=True)
        with comp3: st.image(prew_comp, caption="Prewitt", use_container_width=True)
        with comp4: st.image(otsu_comp, caption="Otsu", use_container_width=True)


        #Tab 5: Morphology & Analysis
    with tab5:
        st.header("Morphological Operations & Object Analysis")
        st.info("Explore Basic Operations (Erosion, Dilation), Advanced Methods (Opening, Closing), and Object Counting.")

        # --- Preprocessing: Convert to Grayscale then Binary ---
        # Morphological operations generally perform best on binary (black and white) images
        if len(original_img.shape) == 3:
            gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = original_img.copy()

        # Apply Otsu's Thresholding to automatically find the optimal threshold value
        _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        st.markdown("### Image Preprocessing")
        invert_bg = st.checkbox("Invert Colors (Objects MUST be White, Background MUST be Black)", value=True)

        if invert_bg:
            binary_img = cv2.bitwise_not(binary_img)

        st.markdown("### Morphological Parameters Control")
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            # Kernel shape selection
            kernel_shape_name = st.selectbox("Kernel Shape", ("Square", "Ellipse", "Cross"))
        with col_ctrl2:
            # Kernel size (must be an odd number for symmetry)
            k_size = st.slider("Kernel Size (Matrix dimensions)", min_value=3, max_value=21, step=2, value=5)
        with col_ctrl3:
            # Number of iterations to apply the morphological operation
            iters = st.slider("Number of Iterations", min_value=1, max_value=5, step=1, value=1)

        # Create the structuring element (kernel) based on user selection
        if kernel_shape_name == "Square":
            shape_type = cv2.MORPH_RECT
        elif kernel_shape_name == "Ellipse":
            shape_type = cv2.MORPH_ELLIPSE
        else:
            shape_type = cv2.MORPH_CROSS

        kernel = cv2.getStructuringElement(shape_type, (k_size, k_size))

        st.markdown("---")

        # --- Section 1: Basic Operations (Erosion & Dilation) ---
        st.subheader("1. Basic Operations: Erosion & Dilation")

        eroded_img = cv2.erode(binary_img, kernel, iterations=iters)
        dilated_img = cv2.dilate(binary_img, kernel, iterations=iters)

        col1_1, col1_2, col1_3 = st.columns(3)
        with col1_1:
            st.image(binary_img, caption="Base Binary Image", use_container_width=True)
            st.caption("*The original image converted to binary (Black & White) to focus on shapes.*")
        with col1_2:
            st.image(eroded_img, caption="Erosion", use_container_width=True)
            st.caption("**Erosion:** Shrinks the boundaries of foreground objects. Used to remove small noises.")
        with col1_3:
            st.image(dilated_img, caption="Dilation", use_container_width=True)
            st.caption("**Dilation:** Expands the boundaries of foreground objects. Used to join broken parts.")

        st.markdown("---")

        # --- Section 2: Advanced Operations (Opening & Closing) ---
        st.subheader("2. Advanced Operations: Opening & Closing")

        opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=iters)
        closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=iters)

        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.image(opened_img, caption="Opening (Erosion ➔ Dilation)", use_container_width=True)
            st.caption("**Opening:** Useful for removing small noise (like salt noise) from the background.")
        with col2_2:
            st.image(closed_img, caption="Closing (Dilation ➔ Erosion)", use_container_width=True)
            st.caption("**Closing:** Useful for closing small holes inside the foreground objects.")

        st.markdown("---")

        # --- Section 3: Object Analysis (Connected Components) ---
        st.subheader("3. Object Counting (Connected Components Analysis)")

        target_img_choice = st.radio(
            "Select image state for object counting:",
            ("Base Binary", "After Opening (Noise Removed)", "After Closing (Gaps Filled)"),
            horizontal=True
        )

        if target_img_choice == "Base Binary":
            target_for_counting = binary_img
        elif target_img_choice == "After Opening (Noise Removed)":
            target_for_counting = opened_img
        else:
            target_for_counting = closed_img

        num_labels, labels_img = cv2.connectedComponents(target_for_counting)
        actual_objects_count = num_labels - 1

        label_hue = np.uint8(179 * labels_img / np.max(labels_img))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img_color = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img_color = cv2.cvtColor(labeled_img_color, cv2.COLOR_HSV2RGB)
        labeled_img_color[label_hue == 0] = 0

        col3_1, col3_2 = st.columns([1, 2])
        with col3_1:
            st.metric(label="Detected Objects Count", value=actual_objects_count)
            st.caption("**Connected Components:** Scans the binary image and assigns a unique color to each distinct, continuous blob of pixels.")
        with col3_2:
            st.image(labeled_img_color, caption="Color-Mapped Objects", use_container_width=True)
else:
    # رسالة تظهر عندما لا يتم رفع أي صورة
    st.warning("Please upload an image from the sidebar to activate the lab modules.")

# --- تذييل الصفحة ---
st.markdown("---")
st.caption("© 2026 Alanoud, Shahad, Lamis, Nada, Yara Lab | Part of Taibah University image processing project")


