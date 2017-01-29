import numpy as np
import matplotlib.pyplot as plt
from peewee import *
from create_database import *
import sys
from query_obj import QueryObj
from coded_matrix import coded_matrix

def compute_efficiency(batteryObj, motorObj, propObj, plot_results = True):

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

    if plot_results:
        fig = plt.figure(figsize = (3.5,3.25), tight_layout = True)
        ax = plt.axes()
        plt.setp(ax.get_xticklabels(), fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)
        plt.hold(True)
        line1, = plt.plot(I, eta, 'b-', label = 'Raw Data')
        # line2, = plt.plot(n_test*60., Ct_test, 'r--', linewidth = 2.0, label = 'Regression')
        # line3, = plt.plot(delta_star, A_tube[2,:], 'g-', linewidth = 2.0, label = 'A_pod = 3.0 $m^2$')
        plt.xlabel('Current (A)', fontsize = 12, fontweight = 'bold')
        plt.ylabel('$\eta$', fontsize = 12, fontweight = 'bold')
        plt.title('5x4 Propeller Thrust Coefficient', fontsize = 10, fontweight = 'bold')
        # plt.xlim(10000, 30000)
        # plt.ylim(7,12)
        plt.savefig('/Users/kennethdecker/Desktop/Grand_Challenge/presentation_plots/5x4_MT-2206_efficiency.png', format = 'png', dpi = 300)

    return np.max(eta)

if __name__ == '__main__':

    db = SqliteDatabase('../database/continuum.db')
    db.connect()

    motor_query = Motor.select().where(Motor.name == 'MT-2206').get()
    # prop_query = Prop.select().where(Prop.diameter == 10.0).get()
    prop_query = Prop.select().where(Prop.name == 'Gemfan 5x4').get()
    bat_query = Battery.select().where(Battery.name == 'Zippy1').get()

    print compute_efficiency(bat_query, motor_query, prop_query)