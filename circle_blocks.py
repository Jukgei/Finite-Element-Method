# coding=utf-8

import taichi as ti
from constants import vec

Block = ti.types.struct(
	id=ti.int32,
	center=vec,
	radius=ti.f32
)

@ti.data_oriented
class CircleBlocks:

	def __init__(self, configs):
		self.blocks_count = len(configs)
		self.blocks = Block.field(shape=self.blocks_count)
		self.init(configs)

	def init(self, block_configs):
		for index in range(len(block_configs)):
			block_config = block_configs[index]
			self.blocks[index].id = block_config.get('id')
			self.blocks[index].center = vec(block_config.get('block_center'))
			self.blocks[index].radius = block_config.get('block_radius')