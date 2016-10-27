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

ntm_2836 = Motor(name = 'NTM PropDrive 28-36', weight = 87.0, kv = 750.0, \
    max_current = 18.0, max_voltage = 15.0, shaft_diameter = None, cad = None)
ntm_2836.save()

mt_1806 = Motor(name = 'MT-1806', weight = 18.0, kv = 2280.0, \
    max_current = None, max_voltage = None, shaft_diameter = 2.0, cad = None)
mt_1806.save()

mt_2206 = Motor(name = 'MT-2206', weight = 32.0, kv = 1900.0, \
    max_current = None, max_voltage = None, shaft_diameter = 3.0, cad = None)
mt_2206.save()

pm_1806 = Motor(name = 'PM-1806', weight = 19.0, kv = 2300.0, \
    max_current = None, max_voltage = None, shaft_diameter = 3.0, cad = None)
pm_1806.save()

rs_2205 = Motor(name = 'RS-2205', weight = 30.0, kv = 2300.0, \
    max_current = None, max_voltage = None, shaft_diameter = 5.0, cad = None)
rs_2205.save()

db.close()
