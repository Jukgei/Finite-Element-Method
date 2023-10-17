# coding=utf-8

import taichi as ti
from constants import dim, mat
import constants
import json


def sys_init(config):
	constants.dim = config.get('dim')
	constants.auto_diff = 1 if config.get('auto_diff') else 0
	constants.use_explicit_method = config.get('use_explicit_method')
	constants.delta_time = config.get('delta_time')
	constants.g_dir = config.get('g_dir')
	# print('dim isss', dim, constants.dim)
	dim = constants.dim
	constants.vec = ti.types.vector(dim, ti.f32)
	constants.mat = ti.types.matrix(dim, dim, ti.f32)
	constants.index = ti.types.vector(dim+1, ti.i32)
	# print(dim, auto_diff, use_explicit_method, g_dir)

def read_config(file_name):
	try:
		with open(file_name, 'r') as f:
			data = json.load(f)
			return data
	except Exception as e:
		print(e)
		print('Parsing config file error')
		exit(3)

