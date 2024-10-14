import numpy as np
import json

u_dict={
"u_heater":40,
"u_normal":15,
"u_window":5}




class Room:
    def __init__(self,rank,dim,adjacent_rooms,boundary,temperature):
        self.rank=rank
        self.dim=dim
        self.adjacent_rooms =adjacent_rooms if adjacent_rooms else {}  # dictionary of adjacent room numbers
        for key,value in temperature.items():
             if value in u_dict:
                setattr(self,key,u_dict[value])
        for key,value in boundary.items():
                setattr(self,key,value)


def get_config_by_rank(config_file, rank):
    # Load the JSON file
    with open(config_file, 'r') as file:
        config_data = json.load(file)

    # Find the parameter set with the matching rank
    for params in config_data['rooms']:
        if params['rank'] == rank:
            return params

    raise ValueError(f"No parameters found for rank: {rank}")

def get_room_list():
    room_list=[]
    for rank in range(3):
        params = get_config_by_rank("ApartmentLayout.json", rank)
        room = Room(**params)
        room_list.append(room)
    return room_list

room_list=get_room_list()
print(room_list[0].top_t)
print(room_list)

