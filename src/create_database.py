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
    data = CharField(default = 'null') #Define data file directory
    cad = CharField(null = True) #Defin cad file directory

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
    cad = CharField() #Define cad file directory

    class Meta:
        database = db #This model is stored in continuum.db database

class QuadParams(Model):
    name = CharField() #Define name of quad copter
    thrust_loading = FloatField() #Define the thrust to weight ratio for that class
    power_loading = FloatField() #Define Power to weight ratio for quad copter class [W/kg]

    class Meta:
        database = db #This model is stored in continuum.db database

class VehicleClass(Model):
    quad = ForeignKeyField(QuadParams) #link this field to the Quad Params Table

    class Meta:
        database = db #This model is stored in continuum.db database


db.connect()
db.create_tables([Battery, \
    Prop, \
    Motor, \
    ESC, \
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
    max_current = 17.0, Rm = .573, shaft_diameter = 3.0, cad = None)
ntm_2826.save()

ntm_2836 = Motor(name = 'NTM PropDrive 28-36', weight = 87.0, kv = 750.0, \
    max_current = 18.0, Rm = .827, shaft_diameter = 4.0, cad = None)
ntm_2836.save()

mt_1806 = Motor(name = 'MT-1806', weight = 18.0, kv = 2280.0, \
    max_current = 12.0, Rm = .388, shaft_diameter = 2.0, cad = None)
mt_1806.save()

mt_2206 = Motor(name = 'MT-2206', weight = 32.0, kv = 1900.0, \
    max_current = 15.0, Rm = .326, shaft_diameter = 3.0, cad = None)
mt_2206.save()

pm_1806 = Motor(name = 'PM-1806', weight = 19.0, kv = 2300.0, \
    max_current = 15.0, Rm = .5867, shaft_diameter = 3.0, cad = None)
pm_1806.save()

rs_2205 = Motor(name = 'RS-2205', weight = 30.0, kv = 2300.0, \
    max_current = 30.0, Rm = .178, shaft_diameter = 3.0, cad = None)
rs_2205.save()


##### Add Props #####

''' Prop addition syntax =>
prop1 = Prop(name = [char], weight = [float], \
            diameter = [float], pitch = [float], cad = [char])

'''
gemfan1 = Prop(name = 'Gemfan1', weight = 3.0, diameter = 5.0, pitch = 3.0,
                data = '../prop_data/5x3.txt', cad = None)
gemfan1.save()

gemfan2 = Prop(name = 'Gemfan2', weight = 3.0, \
            diameter = 5.0, pitch = 4.0, data = '../prop_data/5x4.txt', cad = None)
gemfan2.save()

gemfan3 = Prop(name = 'Gemfan', weight = 3.0, \
            diameter = 5.0, pitch = 4.0, data = '../prop_data/5x4.txt', cad = None)
gemfan3.save()

gemfan4 = Prop(name = 'Gemfan', weight = 3.8, \
            diameter = 6.0, pitch = 3.0, cad = None)
gemfan4.save()

gemfan5 = Prop(name = 'Gemfan', weight = 3.8, \
            diameter = 6.0, pitch = 4.5, cad = None)
gemfan5.save()

gemfan6 = Prop(name = 'Gemfan', weight = 8.5, \
            diameter = 10.0, pitch = 4.5, data = '../prop_data/10x4.7.txt', cad = None)
gemfan6.save()

gemfan7 = Prop(name = 'Gemfan', weight = 8.5, \
            diameter = 12.0, pitch = 4.5, cad = None)
gemfan7.save()

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


db.close()
