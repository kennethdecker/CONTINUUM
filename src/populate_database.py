''' This module is intended to add entries to continuum.db. Code is grouped into sections based on the model being poulated. 
Any database entries made by the users should be reflected in this module for the sake of traceability '''

from peewee import *
from create_database import *
import os

db = SqliteDatabase('../database/continuum.db')

db.connect()

##### Add Motors #####

''' 
motor addition syntax =>
motor1 = Motor(name = [char], weight = [float], kv = [float], \ 
        max_current = [float], max_voltage = [float], \
        shaft_diameter = [float], cad = [char])

'''

ntm_2826 = Motor(name = 'NTM PropDrive 28-26', weight = 54.0, kv = 1000.0, \
    max_current = 15.0, max_voltage = 15.0, shaft_diameter = None, cad = None)
ntm_2826.save()

db.close()
