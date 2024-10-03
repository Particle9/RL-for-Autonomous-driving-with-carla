import carla
import time
import numpy as np
import cv2
import math
import random
import csv
import os
from tqdm import tqdm
import traceback

class CarlaDataCollector:
    def __init__(self, record_duration=60, save_interval=10):
        # Connect to CARLA client
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager(7000)

        # Set up vehicle and sensors
        self.vehicle = None
        self.sensors = []
        self.actors = []  # Store other vehicles and pedestrians
        self.data_buffers = {
            'lidar': [],
            'camera': [],
            'radar_front': [],
            'radar_left': [],
            'radar_right': [],
            'collision': [],
            'lane_invasion': [],
            'control': [],
            'vehicle': []
        }

        # Set the record duration (in seconds)
        self.record_duration = record_duration
        self.save_interval = save_interval  # Save data every X seconds
        self.start_time = time.time()
        self.last_save_time = self.start_time
        self.collision_count = 0
        self.lane_invasion_count = 0

        # Create output directory if it doesn't exist
        if not os.path.exists("carla_data"):
            os.makedirs("carla_data")

        # Create output directories if they don't exist
        os.makedirs("carla_data/images", exist_ok=True)
        os.makedirs("carla_data/lidar", exist_ok=True)
        os.makedirs("carla_data/radar", exist_ok=True)

    def setup_vehicle(self):
        # Destroy the current vehicle if it exists
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()

        # Blueprint library and setup vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]  # Tesla model for example
        spawn_point = random.choice(self.world.get_map().get_spawn_points())  # Choose a random spawn point
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Enable autopilot
        self.vehicle.set_autopilot(True, self.traffic_manager.get_port())  # Enabling autopilot
        print(f"New vehicle spawned with ID: {self.vehicle.id}")

        # Set up sensors for the new vehicle
        self.setup_sensors()

    def setup_sensors(self):
        # Destroy any existing sensors
        for sensor in self.sensors:
            sensor.stop()
            sensor.destroy()
        self.sensors = []

        # Sensor blueprints
        blueprint_library = self.world.get_blueprint_library()

        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('channels', '64')

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '90')

        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '30')
        radar_bp.set_attribute('range', '20')

        collision_bp = blueprint_library.find('sensor.other.collision')
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')

        # Sensor transforms
        lidar_transform = carla.Transform(carla.Location(z=2.5))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        radar_front_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
        radar_side_left_transform = carla.Transform(carla.Location(y=-1.0, z=1.0))
        radar_side_right_transform = carla.Transform(carla.Location(y=1.0, z=1.0))

        # Spawn sensors and assign callbacks
        self.sensors.append(self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle))
        self.sensors[-1].listen(self.lidar_callback)

        self.sensors.append(self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle))
        self.sensors[-1].listen(self.camera_callback)

        self.sensors.append(self.world.spawn_actor(radar_bp, radar_front_transform, attach_to=self.vehicle))
        self.sensors[-1].listen(self.radar_callback_front)

        self.sensors.append(self.world.spawn_actor(radar_bp, radar_side_left_transform, attach_to=self.vehicle))
        self.sensors[-1].listen(self.radar_callback_side_left)

        self.sensors.append(self.world.spawn_actor(radar_bp, radar_side_right_transform, attach_to=self.vehicle))
        self.sensors[-1].listen(self.radar_callback_side_right)

        self.sensors.append(self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle))
        self.sensors[-1].listen(self.collision_callback)

        self.sensors.append(self.world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle))
        self.sensors[-1].listen(self.lane_invasion_callback)

    def spawn_traffic(self, num_vehicles=10, num_pedestrians=10):
        blueprint_library = self.world.get_blueprint_library()

        # Spawn vehicles
        vehicle_bp = blueprint_library.filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()

        for _ in range(num_vehicles):
            spawn_point = random.choice(spawn_points)
            vehicle = self.world.try_spawn_actor(random.choice(vehicle_bp), spawn_point)
            if vehicle:
                vehicle.set_autopilot(True, self.traffic_manager.get_port())  # Enable autopilot for spawned traffic
                self.actors.append(vehicle)

        # Spawn pedestrians
        pedestrian_bp = blueprint_library.filter('walker.pedestrian.*')
        walker_controller_bp = blueprint_library.find('controller.ai.walker')

        for _ in range(num_pedestrians):
            spawn_point = carla.Transform(carla.Location(x=random.uniform(-5, 5), y=random.uniform(-5, 5)))
            pedestrian = self.world.try_spawn_actor(random.choice(pedestrian_bp), spawn_point)
            if pedestrian:
                self.actors.append(pedestrian)
                # Create a controller for the pedestrian to move it
                walker_controller = self.world.try_spawn_actor(walker_controller_bp, carla.Transform(), pedestrian)
                if walker_controller:
                    walker_controller.start()
                    walker_controller.go_to_location(self.world.get_random_location_from_navigation())
                    walker_controller.set_max_speed(1 + random.random())  # Random speed for variety

    # Callback functions for sensors
    def lidar_callback(self, point_cloud):
        timestamp = time.time() - self.start_time
        # Convert point cloud data to NumPy array
        points = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))[:, :3]  # Extract x, y, z

        # Save the point cloud to a file
        filename = f"carla_data/lidar/lidar_{self.vehicle.id}_{timestamp:.2f}.npy"
        np.save(filename, points)

        # Record the filename and timestamp
        self.data_buffers['lidar'].append({
            'timestamp': timestamp,
            'lidar_file': filename,
            'vehicle_id': self.vehicle.id
        })

    def camera_callback(self, image):
        timestamp = time.time() - self.start_time
        # Convert image data to NumPy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]  # Extract RGB

        # Save the image to a file
        filename = f"carla_data/images/image_{self.vehicle.id}_{timestamp:.2f}.png"
        cv2.imwrite(filename, array, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Record the filename and timestamp
        self.data_buffers['camera'].append({
            'timestamp': timestamp,
            'image_file': filename,
            'vehicle_id': self.vehicle.id
        })

        # Show image using OpenCV
        # cv2.imshow("Camera View", array)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     self.shutdown()


    def radar_callback(self, radar_data, radar_position):
        timestamp = time.time() - self.start_time
        radar_points = []
        for detection in radar_data:
            radar_points.append({
                'altitude': detection.altitude,
                'azimuth': detection.azimuth,
                'depth': detection.depth,
                'velocity': detection.velocity
            })

        # Save radar data to a file
        filename = f"carla_data/radar/radar_{radar_position}_{self.vehicle.id}_{timestamp:.2f}.npy"
        np.save(filename, radar_points)

        # Record the filename and timestamp
        self.data_buffers[f'radar_{radar_position}'].append({
            'timestamp': timestamp,
            'radar_file': filename,
            'vehicle_id': self.vehicle.id
        })
    
    def radar_callback_front(self, radar_data):
        self.radar_callback(radar_data, 'front')

    def radar_callback_side_left(self, radar_data):
        self.radar_callback(radar_data, 'left')

    def radar_callback_side_right(self, radar_data):
        self.radar_callback(radar_data, 'right')

    def collision_callback(self, event):
        timestamp = time.time() - self.start_time
        self.data_buffers['collision'].append({'timestamp': timestamp, 'collision_event': event, 'vehicle_id': self.vehicle.id})

    def lane_invasion_callback(self, event):
        timestamp = time.time() - self.start_time
        self.data_buffers['lane_invasion'].append({'timestamp': timestamp, 'lane_invasion_event': event, 'vehicle_id': self.vehicle.id})

    def control_and_vehicle_data(self):
        if not self.vehicle or not self.vehicle.is_alive:
            print("Vehicle actor is destroyed. Spawning a new vehicle.")
            self.setup_vehicle()  # Respawn the vehicle if destroyed
            return False  # Skip this iteration until the new vehicle is spawned

        try:
            # Get control info and vehicle state
            control = self.vehicle.get_control()
            velocity = self.vehicle.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

            proximity = self.check_proximity_to_vehicles()
            fuel_consumption = control.throttle * speed * 0.01  # Simplified consumption
            traffic_light_state = self.vehicle.get_traffic_light().get_state() if self.vehicle.get_traffic_light() else "None"

            # Store control data along with the vehicle ID
            self.data_buffers['control'].append({
                "timestamp": time.time() - self.start_time,
                "throttle": control.throttle,
                "brake": control.brake,
                "steering": control.steer,
                "vehicle_id": self.vehicle.id
            })

            # Store vehicle state data along with the vehicle ID
            self.data_buffers['vehicle'].append({
                "timestamp": time.time() - self.start_time,
                "speed": speed,
                "proximity": proximity,
                "fuel_consumption": fuel_consumption,
                "traffic_light_state": traffic_light_state,
                "vehicle_id": self.vehicle.id
            })
        except RuntimeError as e:
            print(f"Error while accessing vehicle data: {e}")
            return False

        return True  # Data successfully recorded

    def check_proximity_to_vehicles(self):
        proximity_threshold = 10.0  # meters
        proximity = proximity_threshold
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.id != self.vehicle.id:  # Skip ego vehicle
                distance = self.vehicle.get_location().distance(actor.get_location())
                if distance < proximity_threshold:
                    proximity = min(proximity, distance)
        return proximity

    def save_data_periodically(self):
        """Periodically save the data buffers to CSV files and clear memory."""
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            self.last_save_time = current_time

            # Save the buffered data to CSV files
            for data_type, buffer in self.data_buffers.items():
                if buffer:  # Only save if there's data
                    file_path = f"carla_data/{data_type}_data.csv"
                    fieldnames = buffer[0].keys()
                    with open(file_path, mode='a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=fieldnames)
                        if os.stat(file_path).st_size == 0:  # Write headers if file is empty
                            writer.writeheader()
                        writer.writerows(buffer)

                    # Clear the buffer after saving
                    self.data_buffers[data_type].clear()

    def run(self):
        try:
            # Add tqdm progress bar
            with tqdm(total=self.record_duration, desc="Recording Progress") as pbar:
                while time.time() - self.start_time < self.record_duration:
                    # print('A')
                    if not self.control_and_vehicle_data():  # If the vehicle is destroyed, respawn it
                        continue
                    # print('B')
                    self.save_data_periodically()
                    pbar.update(0.1)  # Update the progress bar based on the time step
                    time.sleep(0.1)  # Data collection frequency

        except KeyboardInterrupt:
            self.shutdown()

        self.shutdown()

    def shutdown(self):
        print("Shutting down...")
        for sensor in self.sensors:
            sensor.stop()
            sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        for actor in self.actors:
            actor.destroy()
        cv2.destroyAllWindows()

        # Save any remaining data in the buffers before exiting
        self.save_data_periodically()


if __name__ == '__main__':
    # You can pass the desired record duration (in seconds) as a parameter.
    record_duration = 7200  # Set to 2 minutes for example
    save_interval = 10  # Save every 10 seconds
    collector = CarlaDataCollector(record_duration=record_duration, save_interval=save_interval)
    collector.setup_vehicle()
    collector.setup_sensors()
    collector.spawn_traffic(num_vehicles=15, num_pedestrians=10)  # Spawning traffic
    collector.run()