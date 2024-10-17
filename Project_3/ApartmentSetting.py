import numpy as np
import json

u_dic={
"u_heater":40,
"u_normal":15,
"u_window":5}




class Room:
    def __init__(self,rank,dim,adjacent_rooms,boundary,wall,temp_init=20,h=1/20):
        self.rank=rank
        self.dim=dim
        self.n=int(dim[0]/h)
        self.m=int(dim[1]/h)
        self.adjacent_rooms =adjacent_rooms if adjacent_rooms else {}  # dictionary of adjacent room numbers
        for key,value in wall.items():
             if value in u_dic:
                setattr(self,key,u_dic[value])
        for key,value in boundary.items():
                setattr(self,key,value)
        self.temp_init=temp_init
        self.u=np.ones(np.array(np.array(dim)/h).astype(int))*temp_init
        self.wall_r_left=np.ones([self.n,1])*self.left_w
        self.wall_r_right=np.ones([self.n,1])*self.right_w
        self.wall_r_top=np.ones([1,self.m])*self.top_w
        self.wall_r_bottom=np.ones([1,self.m])*self.bottom_w
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
    for rank in range(4):
        params = get_config_by_rank("ApartmentLayout.json", rank)
        room = Room(**params)
        room_list.append(room)
    return room_list

# room_list=get_room_list()
# print(room_list[3].u)
# print(room_list)
# print(room_list[1].adjacent_rooms["1"]["rank"])
# # uu=room_list[0].u
# # uu[:,0]=np.zeros([20,])
# # print(room_list[0].u)
# print(room_list[1].adjacent_rooms)
# print(room_list[1].u)
# x=room_list[1].u
# x[:,0]=np.zeros([40,])
# print(room_list[1].u)