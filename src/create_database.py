''' Thile module is intended to create the data base that will store information for UAV compontents. 
This module will create a SQLite database using the peewee framework. All information will be stored as .db file.
This module will also define the classes that will be used for each type of component. The model names, field names,
and field data types will all be defined in the class definition. These models will then be added to the SQLite
database and be saved. Other modules will be used to populate the database.

References:
http://docs.peewee-orm.com/en/latest/ '''

from peewee import *
import numpy as np
import os

os.system('rm -rf ../database/continuum.db')

db = SqliteDatabase('../database/continuum.db')


class Battery(Model):
    name = CharField() #Define entry name
    weight = FloatField() #Define Weight in g
    volts = FloatField() #Define voltage
    charge = FloatField() #Define battery charge in mAh
    chemistry = CharField() #Define battery chemistry
    num_cells = IntegerField() #Define number of cells in pack
    cad = CharField(null = True) #Define cad file directory

    class Meta:
        database = db #This model is stored in continuum.db database

class Prop(Model):
    name = CharField() #Define entry name
    weight = FloatField() #Define weight in g
    diameter = FloatField() #Define prop diameter in in.
    pitch = FloatField() # Define prop pitch in in/rev
    shaft_diameter = FloatField(default = 0.0) #Define allowable prop shaft diameter [mm]
    data = CharField(default = 'null') #Define data file directory
    cad = CharField(null = True) #Defin4 cad file directory

    class Meta:
        database = db #This model is stored in continuum.db database

class Motor(Model):
    name = CharField() #Define entry name
    weight = FloatField() #Define weight in g
    kv = FloatField() #Define KV in RPM/V
    max_current = FloatField(null = True) #Define max current in A
    Rm = FloatField() #Define terminal resistance in ohms
    I0 = FloatField(default = 0.0) #Define zero-load current in A
    shaft_diameter = FloatField(null = True) #define shaft diameter in mm
    cad = CharField(null = True) #Define cad file directory

    class Meta:
        database = db #This model is stored in continuum.db database

class ESC(Model):
    name = CharField() #Define entry name
    input_voltage = FloatField() #Define Input voltage
    max_constant_current = FloatField() #Define maximum constant current in A
    max_peak_current = FloatField() #Define maximum peak current in A
    weight = FloatField() #Define weight in g
    cad = CharField(null = True) #Define cad file directory

    class Meta:
        database = db #This model is stored in continuum.db database

class QuadParams(Model):
    name = CharField() #Define name of quad copter
    thrust_loading = FloatField() #Define the thrust to weight ratio for that class
    power_loading = FloatField(null = True) #Define Power to weight ratio for quad copter class [W/kg]

    class Meta:
        database = db #This model is stored in continuum.db database

class Frame(Model):
    name = CharField() #Define name of frame
    weight = FloatField() #Define weight of frame [g]
    total_arm_length = FloatField() #Define total length of arm from center [in]
    prop_distance = FloatField() #Define distance of prop from center [in]
    hole_diameter = FloatField() #Define hole diameter [mm]
    cad = CharField(null = True) #Define cad file directory

    class Meta:
        database = db #This model is stored in continuum.db database

class Controller(Model):
    name = CharField() #Define name of controller
    weight = FloatField() #Define weight of controller [g]
    cad = CharField() #Define cad file directory

    class Meta:
        database = db #This model is stored in continuum.db database

class VehicleClass(Model):
    quad = ForeignKeyField(QuadParams) #link this field to the Quad Params Table

    class Meta:
        database = db #This model is stored in continuum.db database

# if __name__ == '__main__':

db.connect()
db.create_tables([Battery, \
    Prop, \
    Motor, \
    ESC, \
    Frame, \
    Controller, \
    QuadParams, \
    VehicleClass])

##### Add Motors #####

''' 
motor addition syntax =>
motor1 = Motor(name = [char], weight = [float], kv = [float], \ 
        max_current = [float], Rm = [float], I0 = [float], \
        shaft_diameter = [float], cad = [char])

'''

ntm_2826 = Motor(name = 'NTM PropDrive 28-26', weight = 54.0, kv = 1200.0, \
    max_current = 17.0, Rm = .05899, shaft_diameter = 3.0, cad = None)
ntm_2826.save()

ntm_2836 = Motor(name = 'NTM PropDrive 28-36', weight = 87.0, kv = 750.0, \
    max_current = 18.0, Rm = .06236, I0 = .2, shaft_diameter = 4.0, cad = None)
ntm_2836.save()

mt_1806 = Motor(name = 'MT-1806', weight = 18.0, kv = 2280.0, \
    max_current = 12.0, Rm = .04897, I0 = .2, shaft_diameter = 2.0, cad = None)
mt_1806.save()

mt_2206 = Motor(name = 'MT-2206', weight = 32.0, kv = 1900.0, \
    max_current = 15.0, Rm = .05627, I0 = .6, shaft_diameter = 3.0, cad = None)
mt_2206.save()

pm_1806 = Motor(name = 'PM-1806', weight = 19.0, kv = 2300.0, \
    max_current = 15.0, Rm = .05514, shaft_diameter = 3.0, cad = None)
pm_1806.save()

rs_2205 = Motor(name = 'RS-2205', weight = 30.0, kv = 2300.0, \
    max_current = 30.0, Rm = .06320, shaft_diameter = 3.0, cad = None)
rs_2205.save()


##### Add Props #####

''' Prop addition syntax =>
prop1 = Prop(name = [char], weight = [float], \
            diameter = [float], pitch = [float], cad = [char])

'''
gemfan1 = Prop(name = 'Gemfan 5x3', weight = 3.0, diameter = 5.0, pitch = 3.0, shaft_diameter = 3.0, \
                data = '../prop_data/5030.csv', cad = None)
gemfan1.save()

# gemfan2 = Prop(name = 'Gemfan 5x4', weight = 3.0, \
#             diameter = 5.0, pitch = 4.0, data = '../prop_data/5040.csv', cad = None)
# gemfan2.save()

gemfan3 = Prop(name = 'Gemfan 5x4.5', weight = 3.0, \
            diameter = 5.0, pitch = 4.5, data = '../prop_data/5045.csv', cad = None)
gemfan3.save()

gemfan4 = Prop(name = 'Gemfan 6x3', weight = 3.8, \
            diameter = 6.0, pitch = 3.0, data = '../prop_data/6030.csv', cad = None)
gemfan4.save()

gemfan5 = Prop(name = 'Gemfan 6x4.5', weight = 3.8, \
            diameter = 6.0, pitch = 4.5, data = '../prop_data/6045.csv', cad = None)
gemfan5.save()

# gemfan6 = Prop(name = 'Gemfan 10x4.5', weight = 8.5, \
#             diameter = 10.0, pitch = 4.5, data = '../prop_data/10x4_5/', cad = None)
# gemfan6.save()

# gemfan7 = Prop(name = 'Gemfan 12x4.5', weight = 8.5, \
#             diameter = 12.0, pitch = 4.5, data = '../prop_data/12x4_5/', cad = None)
# gemfan7.save()

##### Add Battery #####

''' Battery addition syntax =>

battery1 = Battery(name = [char], weight = [float], volts = [float], \
            charge = [float], chemistry = [char], num_cells = [int], cad = [char])

'''

battery1 = Battery(name = 'Zippy1', weight = 408.0, volts = 11.1, \
            charge = 5000.0, chemistry = 'LiPo', num_cells = 3, cad = None)
battery1.save()

battery2 = Battery(name = 'Multistar1', weight = 185.0, volts = 11.1, \
            charge = 3000.0, chemistry = 'LiPo', num_cells = 3, cad = None)
battery2.save()

battery3 = Battery(name = 'Turnigy1', weight = 200.0, volts = 11.1, \
            charge = 2700.0, chemistry = 'LiPo', num_cells = 3, cad = None)
battery3.save()

battery4 = Battery(name = 'Multistar2', weight = 335.0, volts = 11.1, \
            charge = 5200.0, chemistry = 'LiPo', num_cells = 3, cad = None)
battery4.save()

battery5 = Battery(name = 'Turnigy2', weight = 190.0, volts = 11.1, \
            charge = 2200.0, chemistry = 'LiPo', num_cells = 3, cad = None)
battery5.save()

battery6 = Battery(name = 'Venom1', weight = 435.0, volts = 11.1, \
            charge = 6400.0, chemistry = 'LiPo', num_cells = 3, cad = None)
battery6.save()

battery7 = Battery(name = 'Zippy2', weight = 155.0, volts = 11.1, \
            charge = 2200.0, chemistry = 'LiPo', num_cells = 3, cad = None)
battery7.save()

battery8 = Battery(name = 'Multistar3', weight = 330.0, volts = 14.8, \
            charge = 4000.0, chemistry = 'LiPo', num_cells = 4, cad = None)
battery8.save()

battery9 = Battery(name = 'Zippy3', weight = 205.0, volts = 16.1, \
            charge = 2200.0, chemistry = 'LiPo', num_cells = 4, cad = None)
battery9.save()

##### Add Quad Class info ######
'''  syntax =>

class1 = QuadParams(name = [char], thrust_loading = [float], power_loading = [float])

'''

high_endurance = QuadParams(name = 'High Endurance', thrust_loading = 2.0, power_loading = 75.0)
high_endurance.save()

trainer = QuadParams(name = 'Trainer', thrust_loading = 2.5, power_loading = 75.0)
trainer.save()

sport = QuadParams(name = 'Sport', thrust_loading = 4.0, power_loading = 100.0)
sport.save()

acrobatic = QuadParams(name = 'Acrobatic', thrust_loading = 6.0, power_loading = 125.0)
acrobatic.save()

racing = QuadParams(name = 'Racing', thrust_loading = 8.0, power_loading = 200.0)
racing.save()

##### Add Frame Class Info #####
'''
syntax +>

class1 = Frame(name = '[char], weight = [float], total_arm_length = [float], prop_distance = [float], \
                hole_diameter = [float], cad = [char])
'''
frame150 = Frame(name = 'Frame 150', weight = 154.2, total_arm_length = 150./25.4, prop_distance = 179.87/25.4/np.sqrt(2.), \
                hole_diameter = 9.0, cad = None)
frame150.save()

frame200 = Frame(name = 'Frame 200', weight = 177.8, total_arm_length = 200./25.4, prop_distance = 229.87/25.4/np.sqrt(2.), \
                hole_diameter = 9.0, cad = None)
frame200.save()

frame250 = Frame(name = 'Frame 250', weight = 189.0, total_arm_length = 250./25.4, prop_distance = 279.87/25.4/np.sqrt(2.), \
                hole_diameter = 9.0, cad = None)
frame250.save()

frame300 = Frame(name = 'Frame 300', weight = 225.0, total_arm_length = 300./25.4, prop_distance = 329.87/25.4/np.sqrt(2.), \
                hole_diameter = 9.0, cad = None)
frame300.save()




db.close()
