import numpy as np
import matplotlib.pyplot as plt
from peewee import *
from create_database import *
import sys

def propulsion_characterization(motorObj, batteryObj, propObj, plot = False):

    '''  This function generates characteristic plots for a propulsion system. Produces plots of RPM, power, 
    torque, and efficiency vs current. 

    Inputs
    ------
    motorObj: (type obj)
        Input model instance of motor to be used in propulsion system
    batteryObj:  (type obj)
        Intput model instance of battery to be used in propulsion system
    propObj: (type obj)
        Input model instance of propeller to be used in propulsion system
    plot: (type bool)
        Input to determine whether or not plots are generated

     '''

    #Extract info from model instances
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

    if plot:
        I = np.linspace(1,20)

        torque = Kt*(I-I0)
        RPM = (V-I*Rm)*Kv
        P_out = np.multiply(torque, RPM)/1352.4
        P_in = V*I
        eta = np.divide(P_out,P_in)
        V_pitch = (pitch*RPM*(1.0/720))
        T = .5*rho*(np.pi/4.0)*(d**2.0)*(1.0/144.0)*(V_pitch**2.0)*453.592
        E = ((Q/1000.0)/I)*60.0

        # print V_pitch

        fig1 = plt.figure(tight_layout = True)
        ax1 = plt.axes()
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax4 = ax1.twinx()

        ax3.spines['right'].set_position(('axes', 1.1))
        ax4.spines['right'].set_position(('axes', 1.2))
        ax3.set_frame_on(True)
        ax3.patch.set_visible(False)
        ax4.set_frame_on(True)
        ax4.patch.set_visible(False)

        ax1.plot(I, RPM, 'b-')
        ax2.plot(I, torque, 'r-')
        ax3.plot(I, P_out, 'g-')
        ax4.plot(I, eta, 'm-')
        ax1.set_xlabel('Current (A)', fontsize = 12, fontweight = 'bold')
        ax1.set_ylabel('Motor RPM', color='b', fontsize = 12, fontweight = 'bold')
        ax2.set_ylabel('Motor Torque (oz-in)', color='r', fontsize = 12, fontweight = 'bold')
        ax3.set_ylabel('Power out (W)', color='g', fontsize = 12, fontweight = 'bold')
        ax4.set_ylabel('Efficiency', color='m', fontsize = 12, fontweight = 'bold')
        ax4.set_ylim(bottom = 0.0, top = 1.0)
        plt.show()

        fig2 = plt.figure()
        ax1 = plt.axes()
        ax2 = ax1.twinx()
        ax1.plot(I, T,'b-')
        ax2.plot(I, E, 'r-')
        ax1.set_xlabel('Current (A)', fontsize = 12, fontweight = 'bold')
        ax1.set_ylabel('Thrust (g)', color='b', fontsize = 12, fontweight = 'bold')
        ax2.set_ylabel('Endurance (min)', color='r', fontsize = 12, fontweight = 'bold')
        plt.show()

        # host = host_subplot(111, axes_class=AA.Axes)
        # plt.subplots_adjust(right=0.75)

        # par1 = host.twinx()
        # par2 = host.twinx()

        # offset = 60
        # new_fixed_axis = par2.get_grid_helper().new_fixed_axis
        # par2.axis["right"] = new_fixed_axis(loc="right",
        #                                     axes=par2,
        #                                     offset=(offset, 0))

        # par2.axis["right"].toggle(all=True)


if __name__ == "__main__":
    
    db = SqliteDatabase('../database/continuum.db')
    db.connect()

    motor_query = Motor.select().where(Motor.name == 'MT-1806').get()
    battery_query = Battery.select().where(Battery.name == 'Zippy1').get()
    prop_query = Prop.select().where(Prop.name == 'Gemfan1').get()

    propulsion_characterization(motor_query, battery_query, prop_query, plot = True)

    