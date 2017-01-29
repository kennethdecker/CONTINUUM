import numpy as np
import matplotlib.pyplot as plt
from peewee import *
from create_database import *
import sys
from openpyxl import *
from select_motor_prop import select_motor_prop

def data_validation(file, motorObj, propObj, batteryObj, plot_results = False):

    wb = load_workbook(file, data_only = True)
    sheet_list = wb.get_sheet_names()
    sheet = wb.get_sheet_by_name(sheet_list[0])

    I_col = sheet.columns[7]
    n_col = sheet.columns[8]
    throttle_col = sheet.columns[13]
    T_col = sheet.columns[14]

    I = []
    n = []
    throttle = []
    T = []

    for i in range(len(I_col)):

        if not ((I_col[i].value == None) or isinstance(I_col[i].value, basestring)):
            I.append(float(I_col[i].value))

        if not ((n_col[i].value == None) or isinstance(n_col[i].value, basestring)):
            n.append(float(n_col[i].value))

        if not ((throttle_col[i].value == None) or isinstance(throttle_col[i].value, basestring)):
            throttle.append(float(throttle_col[i].value))

        if not ((T_col[i].value == None) or isinstance(T_col[i].value, basestring)):
            T.append(float(T_col[i].value))

    I_red = []
    n_red = []
    T_red = []

    for i in range(len(I)):

        if throttle[i] > 80.:
            I_red.append(I[i])
            n_red.append(n[i])
            T_red.append(T[i])

    n_pred = []
    I_pred = []

    for i in T_red:
        n_new, I_new, V_new = select_motor_prop(motorObj, propObj, batteryObj, i, plot_results = False)
        n_pred.append(n_new)
        I_pred.append(I_new)

    if plot_results:
        fig = plt.figure(figsize = (3.5,3.25), tight_layout = True)
        ax = plt.axes()
        plt.setp(ax.get_xticklabels(), fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)
        plt.hold(True)
        line1, = plt.plot(T_red, n_red, 'bo', label = 'Raw Data')
        line2, = plt.plot(T_red, n_pred, 'r:', linewidth = 2.0, label = 'Regression')
        # line3, = plt.plot(delta_star, A_tube[2,:], 'g-', linewidth = 2.0, label = 'A_pod = 3.0 $m^2$')
        plt.xlabel('Thrust (gf)', fontsize = 12, fontweight = 'bold')
        plt.ylabel('RPM', fontsize = 12, fontweight = 'bold')
       
        plt.title('5x4 Propeller', fontsize = 10, fontweight = 'bold')
        # plt.xlim(10000, 30000)
        plt.ylim(20000,25000)
        plt.savefig('../Experimental_Data/validation_plots/5x4_prop_rpm.png', format = 'png', dpi = 300)
        plt.show()

        fig2 = plt.figure(figsize = (3.5,3.25), tight_layout = True)
        ax2 = plt.axes()
        plt.setp(ax2.get_xticklabels(), fontsize=8)
        plt.setp(ax2.get_yticklabels(), fontsize=8)
        plt.hold(True)
        line1, = plt.plot(T_red, I_red, 'bo', label = 'Raw Data')
        line2, = plt.plot(T_red, I_pred, 'r:', linewidth = 2.0, label = 'Regression')
        # line3, = plt.plot(delta_star, A_tube[2,:], 'g-', linewidth = 2.0, label = 'A_pod = 3.0 $m^2$')
        plt.xlabel('Thrust (gf)', fontsize = 12, fontweight = 'bold')
        plt.ylabel('Current (A)', fontsize = 12, fontweight = 'bold')
       
        plt.title('5x4 Propeller', fontsize = 10, fontweight = 'bold')
        # plt.xlim(10000, 30000)
        plt.ylim(8,12)
        plt.savefig('../Experimental_Data/validation_plots/5x4_prop_current.png', format = 'png', dpi = 300)
        plt.show()


    return None

if __name__ == '__main__':

    db = SqliteDatabase('../database/continuum.db')
    db.connect()

    # folder = '/Users/kennethdecker/Documents/CONTINUUM/Experimental Data/Final/Small motor/MT 1806-2280KV'
    file = '../Experimental_Data/Final/Small motor/MT 1806-2280KV/5040R_KK_MT1806 2280KV_zippy2200_4S.xlsx'

    motor_query = Motor.select().where(Motor.name == 'MT-1806').get()
    # prop_query = Prop.select().where(Prop.diameter == 10.0).get()
    prop_query = Prop.select().where(Prop.name == 'Gemfan 5x4').get()
    bat_query = Battery.select().where(Battery.name == 'Zippy3').get()

    data_validation(file, motor_query, prop_query, bat_query, plot_results = True)