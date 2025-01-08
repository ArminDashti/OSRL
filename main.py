import pyautogui
from PIL import Image
import time
import os

# Ensure the save directory exists
save_dir = "C:/users/armin/mouse_screenshot"
os.makedirs(save_dir, exist_ok=True)

while True:
    # Get current mouse position
    x, y = pyautogui.position()

    # Define the region for the cropped screenshot
    region_size = 200  # Size of the cropped area (width and height)
    half_size = region_size // 2
    left = x - half_size
    top = y - half_size

    # Capture the screenshot of the defined region
    screenshot = pyautogui.screenshot(region=(left, top, region_size, region_size))

    # Save the screenshot with a timestamped filename
    save_path = os.path.join(save_dir, f"cropped_screenshot_{int(time.time())}.png")
    screenshot.save(save_path)

    print(f"Cropped screenshot saved at: {save_path}")

    # Wait for 1 second before taking the next screenshot
    time.sleep(1)
