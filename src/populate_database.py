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


##### Add Props #####

''' Prop addition syntax =>
prop1 = Prop(name = [char], weight = [float], \
            diameter = [float], pitch = [float], cad = [char])

'''

gemfan1 = Prop(name = 'Gemfan', weight = 3.0, \
            diameter = 5.0, pitch = 30.0, cad = None)
gemfan1.save()

gemfan2 = Prop(name = 'Gemfan', weight = 3.0, \
            diameter = 5.0, pitch = 40.0, cad = None)
gemfan2.save()

gemfan3 = Prop(name = 'Gemfan', weight = 3.0, \
            diameter = 5.0, pitch = 45.0, cad = None)
gemfan3.save()

gemfan4 = Prop(name = 'Gemfan', weight = 3.8, \
            diameter = 6.0, pitch = 30.0, cad = None)
gemfan4.save()

gemfan5 = Prop(name = 'Gemfan', weight = 3.8, \
            diameter = 6.0, pitch = 45.0, cad = None)
gemfan5.save()

gemfan6 = Prop(name = 'Gemfan', weight = 8.5, \
            diameter = 10.0, pitch = 4.5, cad = None)
gemfan6.save()

gemfan7 = Prop(name = 'Gemfan', weight = 8.5, \
            diameter = 12.0, pitch = 4.5, cad = None)
gemfan7.save()

##### Add Battery #####

''' Battery addition syntax =>

battery1 = Battery(name = [char], weight = [float], volts = [float], \
            charge = [float], chemistry = [char], num_cells = [int], cad = [char])

'''

battery1 = Battery(name = 'Zippy', weight = 408.0, volts = 11.1, \
            charge = 5000.0, chemistry = 'LiPo', num_cells = 3, cad = None)
battery1.save()

battery2 = Battery(name = 'Multistar', weight = 185.0, volts = 11.1, \
            charge = 3000.0, chemistry = 'LiPo', num_cells = 3, cad = None)
battery2.save()

battery3 = Battery(name = 'Turnigy', weight = 200.0, volts = 11.1, \
            charge = 2700.0, chemistry = 'LiPo', num_cells = 3, cad = None)
battery3.save()

battery4 = Battery(name = 'Multistar', weight = 335.0, volts = 11.1, \
            charge = 5200.0, chemistry = 'LiPo', num_cells = 3, cad = None)
battery4.save()

battery5 = Battery(name = 'Turnigy', weight = 190.0, volts = 11.1, \
            charge = 2200.0, chemistry = 'LiPo', num_cells = 3, cad = None)
battery5.save()

battery6 = Battery(name = 'Venom', weight = 435.0, volts = 11.1, \
            charge = 6400.0, chemistry = 'LiPo', num_cells = 3, cad = None)
battery6.save()

battery7 = Battery(name = 'Zippy', weight = 155.0, volts = 11.1, \
            charge = 2200.0, chemistry = 'LiPo', num_cells = 3, cad = None)
battery7.save()

battery8 = Battery(name = 'Multistar', weight = 330.0, volts = 14.8, \
            charge = 4000.0, chemistry = 'LiPo', num_cells = 4, cad = None)
battery8.save()

battery9 = Battery(name = 'Zippy', weight = 205.0, volts = 16.1, \
            charge = 2200.0, chemistry = 'LiPo', num_cells = 4, cad = None)
battery9.save()


db.close()
