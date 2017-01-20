from openpyxl import *
import numpy as np
import matplotlib.pyplot as plt
import sys

folder = '/Users/kennethdecker/Documents/CONTINUUM/Experimental Data/Compiled/'
file = 'Compiled_prop_data.xlsx'

out_folder = '/Users/kennethdecker/Documents/CONTINUUM/prop_data/'

wb = load_workbook(folder + file)
sheet_list = wb.get_sheet_names()

for name in sheet_list:
    
    sheet = wb.get_sheet_by_name(name)
    n_col = sheet.columns[9]
    n = []
    for cell in n_col:

        if not ((cell.value == None) or isinstance(cell.value, basestring)):
            n.append(float(cell.value))

    Ct_col = sheet.columns[10]
    Ct = []
    for cell in Ct_col:
        if not ((cell.value == None) or isinstance(cell.value, basestring)):
            Ct.append(float(cell.value))

    n, Ct = (list(t) for t in zip(*sorted(zip(n, Ct))))

    outfile = ''
    for i in range(len(name)):
        if name[i] == '.':
            outfile = outfile + '_'
        else:
            outfile = outfile + name[i]

    f = open(out_folder + outfile + '.txt', 'w')
    f.write('%3s\t%6s\r\n' % ('n', 'CT'))

    for i in range(len(Ct)):
        f.write('%5.3f\t%8.6f\r\n' % (n[i], Ct[i]))

    f.close()

