import struct
import numpy as np

class Memory :
    '''
    Allows access of binary memory data from the ram map.
    Args:
        shm(obj) : A mmap object.
        offset(int) : Bytes to offset for reading memory.
    '''
    def __init__(self,shm,base):
        self.ram = shm
        self.base = base

    def read_flag(self,offset):
        return struct.unpack_from('?', self.ram, self.base + offset)[0]

    def read_u8(self,offset):
        return self.ram[self.base + offset]
    
    def read_s8(self, offset):
        return struct.unpack_from('b', self.ram, self.base + offset)[0]
    
    def read_s16_le(self,offset):
        return struct.unpack_from('<h',self.ram,self.base + offset)[0] 
    
    def read_s16_be(self,offset):
        return struct.unpack_from('>h',self.ram,self.base + offset)[0]
    
    def read_u16_le(self,offset):
        return struct.unpack_from('<H',self.ram,self.base + offset)[0]
    
    def read_u16_be(self,offset):
        return struct.unpack_from('>H',self.ram,self.base + offset)[0]

    def read_s32_le(self,offset):
        return struct.unpack_from('<i',self.ram,self.base + offset)[0]
    
    def read_s32_be(self,offset):
        return struct.unpack_from('>i',self.ram,self.base + offset)[0]
    
    def read_u32_le(self,offset):
        return struct.unpack_from('<I',self.ram,self.base + offset)[0]
    
    def read_u32_be(self,offset):
        return struct.unpack_from('>I',self.ram,self.base + offset)[0]
    

class Pixels :
    '''
    Create rgb array from pixel buffer.

    Args:
        pixel_buffer (obj): A mmap object.
        count (int): Size of pixel_buffer.
        offset (int): Bytes to offset for reading memory.
        height (int): Height of screen.
        width (int): Width of screen.

    '''
    def __init__(self,pixel_buffer,count,offset,height,width):

        self.rgb = np.frombuffer(pixel_buffer,
                                    dtype=np.uint8,
                                    count=count,
                                    offset=offset
                                    ).reshape(height,width,3)
        self.inbattle = False
    
    def update_pixels(self,inbattle) :
        self.inbattle = inbattle



