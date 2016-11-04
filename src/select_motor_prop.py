import numpy as np
import matplotlib.pyplot as plt
from peewee import *
from create_database import *
import sys

def select_motor_prop(motorObj, propObj, T, I_max):
    ''' This function checks to see if a motor-propeller combination is capable of producing the required
    amount of power within maximum current constraints. Prop performance is evaluated using UIUC raw data.
    Assums negligible zero load current

    Inputs
    -----
    motorObj: (type obj)
        Input the model instance of the motor being considered
    propObj: (type obj)
        Input the model instance of the prop being considered
    T: (type float)
        Input the thrust necessary for a single motor. [g]
    P: (type float)
        Input the power necessary for a single motor. [W]
    V: (type float)
        Input battery voltage. [V]
    I_max: (type float)
        Input the maximum allowable current. [A]

    Returns
    -----
    result: (type bool)
        If function returns "True", then the combination meets the constraint. 
        If the functiion returns "False", then the combination does not meet constraint

    References
    -----
    Selig, Michael S. UIUC Propeller Data Site. University of Illinois at Urbana-Champaign, 11/29/15. Web. Accessed 11/3/2016.

     '''

    #Unpack Prop Info
    D = propObj.diameter
    pitch = propObj.pitch
    data_file = propObj.data

    #Unpack Motor Info
    kv = motorObj.kv
    motor_I_max = motorObj.max_current
    kt = 1352.4/kv
    rho = 1.225

    #Evaluate Required current
    if data_file != 'null':
        J, Ct, Cp, eta = np.loadtxt(data_file, skiprows = 1, unpack = True) #Unpack info from raw data
        Ct = Ct[0]
        Cp = Cp[0]

        n_req = np.sqrt(T/(Ct*rho*(D**4.0)))
        P = Cp*rho*(n_req**3.0)*(D**5.0)
        omega_req = n_req*2*np.pi
        tau = P/omega_req
        I_req = tau/kt

        return (I_req < I_max and I_req < motor_I_max)

if __name__ == '__main__':

    db = SqliteDatabase('../database/continuum.db')
    db.connect()

    motor_query = Motor.select().where(Motor.name == 'MT-1806').get()
    prop_query = Prop.select().where(Prop.name == 'Gemfan1').get()

    print select_motor_prop(motor_query, prop_query, 600.0, 15.0)

