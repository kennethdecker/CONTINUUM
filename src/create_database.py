''' Thile module is intended to create the data base that will store information for UAV compontents. 
This module will create a SQLite database using the peewee framework. All information will be stored as .db file.
This module will also define the classes that will be used for each type of component. The model names, field names,
and field data types will all be defined in the class definition. These models will then be added to the SQLite
database and be saved. Other modules will be used to populate the database.

References:
http://docs.peewee-orm.com/en/latest/ '''

from peewee import *
import os

os.system('rm -rf continuum.db')

db = SqliteDatabase('continuum.db')

class Battery(Model):
    weight = FloatField() #Define Weight in lbs
    volts = FloatField() #Define voltage
    charge = FloatField() #Define battery charge in mAh
    chemistry = CharField() #Define battery chemistry
    num_cells = IntegerField() #Define number of cells in pack
    cad = CharField() #Define cad file directory

    class Meta:
        database = db #This model is stored in continuum.db database

class Prop(Model):
    diameter = FloatField() #Define prop diameter in in.
    pitch = FloatField() # Define prop pitch in in/rev
    cad = CharField() #Defin cad file directory

    class Meta:
        database = db #This model is stored in continuum.db database

class Motor(Model):
    weight = FloatField() #Define weight in lbs
    kv = FloatField() #Define KV in RPM/V
    shaft_diamter = FloatField() #define shaft diameter in mm
    cad = CharField() #Define cad file directory

    class Meta:
        database = db #This model is stored in continuum.db database

class ESC(Model):
    input_voltage = FloatField() #Define Input voltage
    max_constant_current = FloatField() #Define maximum constant current in A
    max_peak_current = FloatField() #Define maximum peak current in A
    weight = FloatField() #Define weight in lbs
    cad = CharField() #Define cad file directory

    class Meta:
        database = db #This model is stored in continuum.db database

db.connect()
db.create_tables([Battery, \
    Prop, \
    Motor, \
    ESC])

db.close()
