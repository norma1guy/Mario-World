from typing import Optional

import posix_ipc,mmap,struct,torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Categorical,Bounded,Composite,UnboundedDiscrete,UnboundedContinuous
import numpy as np
from Memory import Memory, Pixels


class MarioEnvironment(EnvBase) :

    def __init__(self,height,width,num_proc,seed=None,device='cuda'):

        super().__init__(device=device)

        if seed is None:
            seed = torch.empty((),dtype=torch.int64).random_().item()

        self.set_seed(seed)
        self.shm = posix_ipc.SharedMemory(f'/RAMP_MAP{num_proc}')
        self.pyFlag = posix_ipc.Semaphore(f'/py_to_lua{num_proc}')
        self.luaFlag = posix_ipc.Semaphore(f'/lua_to_py{num_proc}')
        self.ramSize = 2 * 1024
        self.pixelCount = height * width
        self.pixelSize = self.pixelCount * 3
        self.mm = mmap.mmap(self.shm.fd,8 + self.ramSize + self.pixelSize)
        self.shm.close_fd()
        self.input = struct.unpack('I',self.mm[:4])
        self.ram = Memory(self.mm,8)
        self.pixels = Pixels(self.mm,self.pixelSize,8 + self.ramSize,height,width)


    def _make_spec(self):

        self.observation_spec = Composite(

        )

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def calc_reward(self):

        return


    def _step(self):

        return


    def _reset(self):

        return

    def _get_button_press(self):
        # 0 = No button
        # 0x40 = A
        # 0x80 = B
        # 0xC0 = Both

        return self.ram.read_u8(0x000a)

    def _get_player_mov_dir(self):
        # 1 = right
        # 2 = left
        return self.ram.read_u8(0x0045)

    def _get_player_face_dir(self):
        return self.ram.read_u8(0x0033)

    def _get_player_speed(self):
        # 0xd8 to 0 for left
        # 0 to 0x28 for right
        return self.ram.read_u8(0x0058)

    def _get_player_x_pos(self):
        return self.ram.read_u8(0x006d)

    def _get_player_x_pos_screen(self):
        return self.ram.read_u8(0x0086)

    def _get_player_y_pos(self):
        return self.ram.read_u8(0x00ce)

    def _screen_edge_x_pos(self):
        return self.ram.read_u8(0x0071c)


    def _get_player_state(self):

        '''
        0 = leftmost of screen
        1 = climbing vine
        2 = entering reversed l pipe
        3 = going down a pipe
        4,5 = autowalk
        6 = player dies
        7 = Entering area
        8 = normal
        9 = transform small to large
        10 = large to small
        11 = dying
        12 = fire mario
        '''

        return self.ram.read_u8(0x000e)

    def _get_active_enemies(self):
        #Array of size 5 which tells if enemy is present
        return [self.ram.read_u8(0x000f + i) for i in range(5)]
    
    def _get_enemy_heading_dir(self):
        return [self.ram.read_u8(0x0046 + i) for i in range(5)]

    def _get_enemy_x_pos(self):
        return [self.ram.read_u8(0x006e +i) for i in range(5)]

    def _get_enemy_x_pos_screen(self):
        return [self.ram.read_u8(0x0087 + i) for i in range(5)]

    def _get_enemy_y_pos_screen(self):
        return [self.ram.read_u8(0x00cf + i) for i in range(5)]

    
    def _get_player_info(self):
        mov_dir = self.ram.read_u8(0x0045)
        face_dir = self.ram.read_u8(0x0033)
        speed = self.ram.read_u8(0x0057)
        lives = self.ram.read_u8(0x075a)
        coins = self.ram.read_u8(0x075e)
        world = self.ram.read_u8(0x075f)
        level = self.ram.read_u8(0x0760)

        button_ab = self.ram.read_u8(0x000a)
        state = self.ram.read_u8(0x001d)
        y_pos = self.ram.read_u8(0x00ce)
        x_pos_offset = self.ram.read_u8(0x03ad)




        return

    def _get_powerup_info(self):
        is_present = self.ram.read_u8(0x001b)
        type = self.ram.read_u8(0x0039)
        state = self.ram.read_u8(0x0756)

    def _get_powerup_x_pos_screen(self):
        return self.ram.read_u8(0x008c)

    def _get_powerup_y_pos(self):
        return self.ram.read_u8(0x00d4)
        


    def _get_enemy_info(self):

        enemy_present = [self.ram.read_u8(0x000f + i) for i in range(5)]
        enemy_types = [self.ram.read_u8(0x0016 + i) for i in range(5)]
        dir = self.ram.read_u8(0x0046)
        speed = self.ram.read_u8(0x0058)
        y_pos = [self.ram.read_u8(0x00cf + i) for i in range(5)]



    


    def get_state(self):

        # Player information
        

        # Enemy in



        