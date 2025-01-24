import numpy as np
import pyautogui
import cv2
import time
from pynput.mouse import Listener
import os
import pickle

class DesktopEnv:
    def __init__(self, max_steps=200, region_size=10, threshold=20, save_dir=None):
        self.max_steps = max_steps
        self.region_size = region_size
        self.threshold = threshold
        self.mouse_pos = [500, 500]
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.done = False
        self.observation_image = self._capture_screenshot()
        return self.observation_image

    def _capture_screenshot(self):
        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        resized_screenshot = cv2.resize(screenshot, (64, 64))
        normalized_screenshot = resized_screenshot.astype(np.float32) / 255.0
        if self.save_dir:
            filename = f"screenshot_{int(time.time() * 1000)}.png"
            filepath = os.path.join(self.save_dir, filename)
            cv2.imwrite(filepath, resized_screenshot * 255)
        return normalized_screenshot

    def draw_cursor_box(self, image, position):
        top_left = (position[0] - 10, position[1] - 10)
        bottom_right = (position[0] + 10, position[1] + 10)
        color = (255, 0, 0)
        thickness = 2
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

    def step(self, click_action, predicted_coords):
        self.current_step += 1
        self.mouse_pos = self._update_mouse_position(predicted_coords)
        pyautogui.moveTo(*self.mouse_pos)
        if click_action == 0:
            pyautogui.rightClick()
        elif click_action == 1:
            pyautogui.leftClick()
        elif click_action == 2:
            pyautogui.doubleClick()
        next_observation_image = self._capture_screenshot()
        self.draw_cursor_box(next_observation_image, self.mouse_pos)
        return next_observation_image, self.done, {}

    def _update_mouse_position(self, predicted_coords):
        raw_x, raw_y = predicted_coords[0]
        new_x = int((raw_x // self.region_size) * self.region_size + self.region_size / 2)
        new_y = int((raw_y // self.region_size) * self.region_size + self.region_size / 2)
        return [new_x, new_y]

    def generate_human_trajectories(self, time_limit=5):
        print("Human trajectory generation started. Perform actions.")
        trajectories = []
        self.reset()
        start_time = time.time()
        while time.time() - start_time < time_limit:
            current_state = self.observation_image
            text_query = self._get_text_query()
            x, y = pyautogui.position()
            click_action = self._get_click_action()
            if click_action is None:
                continue
            self.step(click_action, [[x, y]])
            trajectories.append((current_state, text_query, click_action, self.mouse_pos))
            if self._check_exit_condition():
                break
        print("Human trajectory generation finished.")
        return trajectories

    def _get_text_query(self):
        print("What do you want to do? Provide a text query for BERT encoding:")
        return input("Enter text query: ").split()

    def _get_click_action(self):
        try:
            return int(input("Enter click action (0-3): "))
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 3.")
            return None

    def _check_exit_condition(self):
        return input("Press Enter to continue or 'q' to finish: ").lower() == 'q'

    def human_collector(self, time_limit=30, save_dir=None):
        if not save_dir:
            raise ValueError("The parameter 'save_dir' is required and cannot be None.")
        print("Starting human interaction collection.")
        interactions = []
        os.makedirs(save_dir, exist_ok=True)
        self.reset()
        mouse_positions = []
        mouse_clicks = []
        def on_move(x, y):
            mouse_positions.append((x, y))
        def on_click(x, y, button, pressed):
            if pressed:
                mouse_clicks.append((x, y, button.name))
        listener = Listener(on_move=on_move, on_click=on_click)
        listener.start()
        start_time = time.time()
        while time.time() - start_time < time_limit:
            current_state = self._capture_screenshot()
            current_mouse_pos = mouse_positions[-1] if mouse_positions else (0, 0)
            click_action = mouse_clicks.pop(0)[2] if mouse_clicks else "nothing"
            time.sleep(0.1)
            next_mouse_pos = mouse_positions[-1] if mouse_positions else (0, 0)
            next_state = self._capture_screenshot()
            action = [next_mouse_pos[0] - current_mouse_pos[0], next_mouse_pos[1] - current_mouse_pos[1]]
            interactions.append((current_state, action, next_state, click_action, 'exploration'))
            print(f"Action logged: {action}, Click action: {click_action}")
        listener.stop()
        file_path = os.path.join(save_dir, "human_interactions.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(interactions, f)
        print(f"Interactions saved to {file_path}")
        print("Human interaction collection finished.")
        return interactions

env = DesktopEnv()
env.human_collector(save_dir='c:/users/armin/envenv/')
