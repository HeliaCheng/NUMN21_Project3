import numpy as np
import json

u_heater=40;
u_normal=15;
u_window=5;

class Room:
    def __init__(self, dimensions,boundary, adjacent_rooms=None):
        self.dimensions = dimensions
        self.adjacent_rooms = adjacent_rooms if adjacent_rooms else {}  # dictionary of adjacent room numbers

        # self.top=top;
        # self.bottom=bottom;
        # self.left=left;
        # self.right=right;
        # self.temperature = np.full(self.dimensions, self.u_normal)
        # self.teperature[0,:]=np.full([1,self.dimensions[1]],self.top)
        # self.temperature[-1,:]=np.full([1,self.dimensions[1]],self.bottom)
        # self.temperature[:,0]=np.full([1,self.dimensions[1]],self.left)
        # self.temperature[:, -1] = np.full([1, self.dimensions[1]], self.right)

def load_apartment_layout(filename='ApartmentLayout.json'):

    try:
        with open(filename, 'r') as f:
            apart_config = json.load(f)

        # Ensure that the 'rooms' key is present in the configuration
        if 'rooms' not in apart_config:
            raise KeyError(f"Expected key 'rooms' not found in the JSON file {filename}")

        rooms = [Room(**room_config) for room_config in apart_config['rooms']]

        return rooms

    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} was not found.")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Invalid JSON format in {filename}.")