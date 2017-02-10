import numpy as np
from peewee import *
from create_database import *
import sys
from coded_obj import CodedObj
from query_obj import QueryObj

class DesVars(object):
    def __init__(self, endurance, TqW):
        ''' 
        endurance: (float) [min]
        TqW: (float) [unitless]

        '''
        self.endurance = endurance
        self.TqW = TqW

def main():
    #Define design variables
    desVars = DesVars(endurance = 10., TqW = 2.0)

    #Query Database
    db = SqliteDatabase('../database/continuum.db')
    db.connect()

    motor_query = Motor.select()#.where(Motor.name == 'MT-1806').get()
    battery_query = Battery.select()#.where(Battery.name == 'Zippy1').get()
    prop_query = Prop.select()#.where(Prop.name == 'Gemfan1').get()
    frame_query = Frame.select()

    data_set = QueryObj(battery_query, motor_query, prop_query, frame = frame_query)

    #Set full factorial study values
    level_array = np.array([len(battery_query), len(motor_query), len(prop_query), len(frame_query)], dtype = np.int)
    module_list = ['Battery', 'Motor', 'Prop', 'Frame']

    design_array = CodedObj(level_array, module_list, data_set)

    #Set TOPSIS parameters
    attribute_list = ['Weight', 'Efficiency', 'Current', 'Endurance']
    decision_array = np.array([0, 1, 0, 1])
    scaling_array = np.array([.5, .1, .2, .2])
    print design_array.namelist

if __name__ == '__main__':

    main()