import numpy as np
import matplotlib.pyplot as plt
from peewee import *
from create_database import *
import sys

class QueryObj(object):
    def __init__(self, battery, motor, prop, esc = None, frame = None):
        self.battery = battery
        self.motor = motor
        self.prop = prop
        self.esc = esc
        self.frame = frame