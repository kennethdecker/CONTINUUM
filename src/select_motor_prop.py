import numpy as np
import matplotlib.pyplot as plt
from peewee import *
from create_database import *
import sys

def select_motor_prop(motorObj, propObj, batObj, T, I_max = None, plot_results = True):
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

        n_new = 200.0
        Ct = 1.0
        eps = 1.

        i = 1
        while eps > 1e-6:
            
            n_old = n_new
            # Ct = p(n_old)
            # n_new = np.sqrt(T/(rho*Ct*((D/12.)**4.0)))
            # lam = .8
            # n_new = lam*n_new + (1-lam)*n_old

            h = 1e-4
            f1 = p(n_old) - (T/(rho*(n_old**2.)*((D/12.)**4.)))
            n_h = n_old + h
            f2 =  p(n_h) - (T/(rho*(n_h**2.)*((D/12.)**4.)))
            f_prime = (f2-f1)/h

            n_new = n_old - (f1/f_prime)

            eps = np.abs((n_new - n_old)/n_old)

            i += 1
            # print 'n_new = %f' % n_new
            # print 'Ct = %f' % Ct

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
        eta = np.sqrt((2.*((T/101.971621298)**3.))/(1.225*((D*.0254)**2.)))/((V_in - I_req*Rm)*I_req)
        print 'RPM = %f' % rpm

        print 'Current = %f' % I_req
        print 'Voltage = %f' % V_req

        if plot_results:

            n_test = np.linspace(150, 500, 5)
            Ct_test = map(lambda x: p(x), n_test)
            T_list = map(lambda Ct, n: Ct*(rho*(n**2.)*((D/12.)**4.)), Ct_list, n_list)
            Cq_test = map(lambda y: p2(y), n_list)

            fit3 = np.polyfit(n_list, T_list, 2)
            p3 = np.poly1d(fit3)
            T_test = map(lambda x: p3(x), n_test)

            fig = plt.figure(figsize = (3.5,3.25), tight_layout = True)
            ax = plt.axes()
            plt.setp(ax.get_xticklabels(), fontsize=8)
            plt.setp(ax.get_yticklabels(), fontsize=8)
            plt.hold(True)
            line1, = plt.plot(n_list*60., Ct_list, 'bo', label = 'Raw Data')
            line2, = plt.plot(n_test*60., Ct_test, 'r--', linewidth = 2.0, label = 'Regression')
            # line3, = plt.plot(delta_star, A_tube[2,:], 'g-', linewidth = 2.0, label = 'A_pod = 3.0 $m^2$')
            plt.xlabel('RPM', fontsize = 12, fontweight = 'bold')
            plt.ylabel('$C_{T}$', fontsize = 12, fontweight = 'bold')
            plt.title('5x4 Propeller Thrust Coefficient', fontsize = 10, fontweight = 'bold')
            plt.xlim(10000, 30000)
            plt.ylim(7,12)
            plt.savefig('/Users/kennethdecker/Desktop/Grand_Challenge/presentation_plots/5x4_prop_thrust.png', format = 'png', dpi = 300)
            # plt.show()

            fig2 = plt.figure(figsize = (3.5,3.25), tight_layout = True)
            ax2 = plt.axes()
            plt.setp(ax2.get_xticklabels(), fontsize=8)
            plt.setp(ax2.get_yticklabels(), fontsize=8)
            plt.hold(True)
            line1, = plt.plot(n_list2*60., Cq_list, 'bo', label = 'Raw Data')
            line2, = plt.plot(n_test*60., Cq_test, 'r--', linewidth = 2.0, label = 'Regression')
            # line3, = plt.plot(delta_star, A_tube[2,:], 'g-', linewidth = 2.0, label = 'A_pod = 3.0 $m^2$')
            plt.xlabel('RPM', fontsize = 12, fontweight = 'bold')
            plt.ylabel('$C_{Q}$', fontsize = 12, fontweight = 'bold')
            plt.title('5x4 Propeller Thrust Coefficient', fontsize = 10, fontweight = 'bold')
            plt.xlim(18000, 30000)
            plt.ylim(1,1.5)
            plt.savefig('/Users/kennethdecker/Desktop/Grand_Challenge/presentation_plots/5x4_prop_torque.png', format = 'png', dpi = 300)
            # plt.show()

            fig3 = plt.figure(figsize = (3.5,3.25), tight_layout = True)
            ax3 = plt.axes()
            plt.setp(ax3.get_xticklabels(), fontsize=8)
            plt.setp(ax3.get_yticklabels(), fontsize=8)
            plt.hold(True)
            line1, = plt.plot(n_list*60., T_list, 'bo', label = 'Raw Data')
            line2, = plt.plot(n_test*60., T_test, 'r--', linewidth = 2.0, label = 'Regression')
            # line3, = plt.plot(delta_star, A_tube[2,:], 'g-', linewidth = 2.0, label = 'A_pod = 3.0 $m^2$')
            plt.xlabel('RPM', fontsize = 12, fontweight = 'bold')
            plt.ylabel('T (gf)', fontsize = 12, fontweight = 'bold')
            plt.title('5x4 Propeller Thrust', fontsize = 10, fontweight = 'bold')
            # plt.xlim(10000, 30000)
            # plt.ylim(.05, .1)
            plt.savefig('/Users/kennethdecker/Desktop/Grand_Challenge/presentation_plots/5x4_prop_raw_thrust.png', format = 'png', dpi = 300)
            # plt.show()

        # return (I_req< I_max) and (V_req < V_in)
        return (rpm, I_req, V_req, eta)
        


if __name__ == '__main__':

    db = SqliteDatabase('../database/continuum.db')
    db.connect()

    motor_query = Motor.select().where(Motor.name == 'MT-1806').get()
    # prop_query = Prop.select().where(Prop.diameter == 10.0).get()
    prop_query = Prop.select().where(Prop.name == 'Gemfan 5x4').get()
    bat_query = Battery.select().where(Battery.name == 'Turnigy1').get()

    print select_motor_prop(motor_query, prop_query, bat_query, 150.0, 15.0, plot_results = False)

