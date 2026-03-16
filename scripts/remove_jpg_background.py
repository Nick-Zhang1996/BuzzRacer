import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def process_images(folder_path="."):
    # 1. Find all *.jpg files in the specified folder
    search_path = os.path.join(folder_path, "*.jpg")
    image_files = glob.glob(search_path)

    if not image_files:
        print("No .jpg files found in the specified folder.")
        return

    print(f"Found {len(image_files)} image(s). Starting processing...\n")

    for img_path in image_files:
        print(f"Processing: {img_path}")

        # Read the image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}, skipping.")
            continue

        # OpenCV reads in BGR format, convert to RGB for Matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the image to HSV to better isolate the green color
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for green in HSV space.
        # (You may need to tweak these slightly depending on your specific shade of green)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        # Create a mask where green pixels are white (255) and everything else is black (0)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # 3. Remove background only on the exterior ring
        # Invert the green mask to get the foreground (the main object)
        foreground_mask = cv2.bitwise_not(green_mask)

        # Find the external contours of the foreground.
        # This acts as a boundary wrapper, ignoring any green gaps inside the object itself.
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank mask and fill in the external contours solidly
        solid_alpha_mask = np.zeros_like(foreground_mask)
        cv2.drawContours(solid_alpha_mask, contours, -1, 255, thickness=cv2.FILLED)

        # Split the original image into its Blue, Green, and Red channels
        b, g, r = cv2.split(img)

        # Merge the channels back together with our solid mask acting as the Alpha (transparency) channel
        # 0 = transparent (exterior green), 255 = opaque (the main object)
        rgba = cv2.merge([b, g, r, solid_alpha_mask])

        # Convert RGBA to RGB+Alpha for Matplotlib display
        rgba_display = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)

        # 2 & 4. Display the original and processed images side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.canvas.manager.set_window_title(f"Processing: {os.path.basename(img_path)}")

        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(rgba_display)
        axes[1].set_title("Processed (Transparent Background)")
        axes[1].axis('off')

        plt.tight_layout()

        # Display the plot. The script will pause here until you close the window.
        # 6. Once closed, it will save and move to process the next image.
        print("Close the image window to save and move to the next file...")
        plt.show()

        # 5. Save the image in png format, maintaining the alpha channel
        base_name = os.path.splitext(img_path)[0]
        out_path = base_name + ".png"

        # OpenCV's imwrite automatically handles the 4-channel RGBA matrix to save transparency
        cv2.imwrite(out_path, rgba)
        print(f"Saved: {out_path}\n")

    print("All images processed successfully.")


if __name__ == "__main__":
    # Run the script on the current directory.
    # Change "." to your folder path (e.g., "C:/images") if needed.
    process_images("./assets/car_imgs/")
