import numpy as np
import openpyxl as xl
import os
import sys

def sort_data():

    root_path = '../Experimental_Data/Final/Small_motor/'
    motor_types = os.listdir(root_path)
    motor_types.remove('.DS_Store')

    i = 0
    for mot in motor_types:
        i += 1
        file_list = os.listdir(root_path + mot + '/')

        for name in file_list:

            file_name = root_path + mot + '/' + name 

            try:
                wb = xl.load_workbook(file_name)
            except:
                print '%s failed to load' % name

        if i > 0:
            break

if __name__ == '__main__':

    sort_data()

