import numpy as np
import matplotlib.pyplot as plt
from peewee import *
from create_database import *
import sys
from query_obj import QueryObj
from coded_matrix import coded_matrix


def evaluate_cases(QueryObj, coded_array, level_array, module_list):

    m,n = coded_array.shape
    attributes = 2
    results = np.zeros((m, attributes), dtype = np.float64)
    modules = len(level_array)

    if np.sum(level_array) != n:
        raise ValueError('Sum of level array must equal number of clumns of coded matrix')

    for i in range(m):

        row = coded_array[i]

        weight = 0
        col = 0

        for j in range(modules):

            num = level_array[j]

            for k in range(num):

                if module_list[j] == 'Battery':
                    weight = weight + coded_array[i][col] * QueryObj.battery[k].weight

                    if coded_array[i][col] == 1:
                        battery = QueryObj.battery[k]

                elif module_list[j] == 'Motor':
                    weight = weight + coded_array[i][col] * QueryObj.motor[k].weight
                    
                    if coded_array[i][col] == 1:
                        motor = QueryObj.motor[k]

                elif module_list[j] == 'Prop':
                    weight = weight + coded_array[i][col] * QueryObj.prop[k].weight
                    
                    if coded_array[i][col] == 1:
                        prop = QueryObj.prop[k]

                elif module_list[j] == 'ESC':
                    weight = weight + coded_array[i][col] * QueryObj.esc[k].weight

                elif module_list[j] == 'Frame':
                    weight = weight + coded_array[i][col] * QueryObj.frame[k].weight

                col = col + 1

        efficiency = compute_efficiency(battery, motor, prop)

        results[i][0] = weight
        results[i][1] = efficiency

    return results

def compute_efficiency(batteryObj, motorObj, propObj):

    Kv = motorObj.kv
    Rm = motorObj.Rm
    I0 = motorObj.I0
    I_max = motorObj.max_current

    V = batteryObj.volts
    Q = batteryObj.charge

    d = propObj.diameter
    pitch = propObj.pitch

    rho = .002377
    Kt = 1352.4/Kv


    I = np.linspace(1,20)

    torque = Kt*(I-I0)
    RPM = (V-I*Rm)*Kv
    P_out = np.multiply(torque, RPM)/1352.4
    P_in = V*I
    eta = np.divide(P_out,P_in)

    return np.max(eta)

if __name__ == "__main__":
    
    db = SqliteDatabase('../database/continuum.db')
    db.connect()

    motor_query = Motor.select()#.where(Motor.name == 'MT-1806').get()
    battery_query = Battery.select()#.where(Battery.name == 'Zippy1').get()
    prop_query = Prop.select()#.where(Prop.name == 'Gemfan1').get()

    data_set = QueryObj(battery_query, motor_query, prop_query)

    level_array = np.array([len(battery_query), len(motor_query), len(prop_query)], dtype = np.int)
    coded_array = coded_matrix(level_array)
    module_list = ['Battery', 'Motor', 'Prop']

    print evaluate_cases(data_set, coded_array, level_array, module_list)

