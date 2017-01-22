import numpy as np
import matplotlib.pyplot as plt
from peewee import *
from create_database import *
import sys

def select_motor_prop(motorObj, propObj, batObj, T, I_max):
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
    data_folder = propObj.data

    #Unpack Motor Info
    kv = motorObj.kv
    motor_I_max = motorObj.max_current
    Rm = motorObj.Rm
    I0 = motorObj.I0
    kt = 1352.4/kv
    rho = .002378

    #Unpack Battery info
    V_in = batObj.volts

    #Evaluate Required current
    # if data_file != 'null':
    #     J, Ct, Cp, eta = np.loadtxt(data_file, skiprows = 1, unpack = True) #Unpack info from raw data
    #     Ct = Ct[0]
    #     Cp = Cp[0]

    #     n_req = np.sqrt(T/(Ct*rho*(D**4.0)))
    #     P = Cp*rho*(n_req**3.0)*(D**5.0)
    #     omega_req = n_req*2*np.pi
    #     tau = P/omega_req
    #     I_req = tau/kt

    #     return (I_req < I_max and I_req < motor_I_max)

    if data_folder != 'null':
        n_list, Ct_list = np.loadtxt(data_folder + 'n_vs_ct.txt', skiprows = 1, unpack = True) #Unpack info from raw data
        fit1 = np.polyfit(n_list, Ct_list, 1)
        p = np.poly1d(fit1)

        n_new = 5000.0
        Ct = 1.0

        eps = 1.

        i = 1
        while eps > 1e-6:
            
            n_old = n_new
            Ct = p(n_old)
            n_new = np.sqrt(T/(rho*Ct*((D/12.)**4.0)))

            eps = np.abs((n_new - n_old)/n_old)
            i += 1

            if i > 100:
                print 'max iter reached'
                print 'n = %f ' % n_new
                print 'esps = %f' % eps
                sys.exit()


        n_list2, Cq_list = np.loadtxt(data_folder + 'n_vs_cq.txt', skiprows = 1, unpack = True)
        fit2 = np.polyfit(n_list2, Cq_list, 1)
        p2 = np.poly1d(fit2)

        Cq = p2(n_new)
        tau = Cq*rho*(n_new**2.0)*((D/12.)**5.)
        rpm = n_new*60.

        I_req = (tau/kt) + I0
        V_req = (rpm/kv) + I_req*Rm
        print rpm

        print I_req
        print V_req

        return (I_req< I_max) and (V_req < V_in)
        


if __name__ == '__main__':

    db = SqliteDatabase('../database/continuum.db')
    db.connect()

    motor_query = Motor.select().where(Motor.name == 'MT-1806').get()
    prop_query = Prop.select().where(Prop.name == 'Gemfan1').get()
    bat_query = Battery.select().where(Battery.name == 'Zippy1').get()

    print select_motor_prop(motor_query, prop_query, bat_query, 200.0, 15.0)

