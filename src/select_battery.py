import numpy as np
import matplotlib.pyplot as plt
from peewee import *
from create_database import *
import sys

def select_battery(batteryObj, endurance, power, SF = .71):

    ''' This function evaluates whether or not a battery is capable of producing the required power for 
    The required amount of time.

    Inputs
    -----
    batteryObj: (type obj)
        Input model instance of battery being evaluated
    endurance: (type float)
        input the necessary endurance of the battery. [min]
    power: (type float)
        input the power the battery must produce. [W]
    SF: (type float)
        input the safety factor for battery discharge. Must be less than 1. Default value is .71

    Returns
    -----
    select_battery: (type bool)
        Returns "True" if battery meets requirement. Returns "False" if battery does not meet requirement

    '''

    V = batteryObj.volts
    Q = batteryObj.charge*(60.0/1000.0)
    I = power/V

    return (endurance/SF) < (Q/I)

if __name__ == '__main__':

    db = SqliteDatabase('../database/continuum.db')
    db.connect()

    battery_query = Battery.select().where(Battery.name == 'Zippy1').get()

    print select_battery(battery_query, 5.0, 222.0)

