import numpy as np


def apply_compatibility(levels, modules, namelist, incompatible, self_compatible = True):

        main_matrix = np.ones((np.sum(levels), (np.sum(levels))), dtype = np.int)
        design = coded_matrix(levels)
        m,n = design.shape
        print 'initial number of cases = %d' % m

        if self_compatible:
            for i in range(len(levels)):
                base = np.sum(levels[:i])
                box_len = levels[i]
                main_diag = [1]*box_len
                diag_mat = np.zeros((box_len, box_len), dtype = np.int)
                np.fill_diagonal(diag_mat, main_diag)

                main_matrix[base:base + box_len, base:base + box_len] = diag_mat

        for i in range(len(incompatible)):

            name1, name2 = incompatible[i]

            for mod1 in modules:
                if name1 in namelist[mod1]:
                    j1 = namelist[mod1].index(name1)

                    break

            for mod2 in modules:
                if name2 in namelist[mod2]:
                    j2 = namelist[mod2].index(name2)
                    break

            mod_num1 = modules.index(mod1)
            mod_num2 = modules.index(mod2)

            base1 = np.sum(levels[:mod_num1])
            base2 = np.sum(levels[:mod_num2])

            main_matrix[base1 + j1][base2 + j2] = 0
            main_matrix[base2 + j2][base1 + j1] = 0

        # self.compatibility = main_matrix
        index_list = []
        for i in range(len(main_matrix)):
            for j in range(i, len(main_matrix[i])):
                val = main_matrix[i][j]
                if val == 0:
                    index_list.append([i,j])
        
        remove = []
        for pair in index_list:
            val1, val2 = pair
            i = 0
            for row in design:
                if (row[val1] == 1 and row[val2] == 1):
                    remove.append(i)
                i += 1

        new_design = np.delete(design, remove, axis = 0)
        
        return new_design

def coded_matrix(levels):

    m = np.prod(levels)
    n = np.sum(levels)
    design = np.zeros((m,n))

    categories = len(levels)

    ssize = m
    ncycles = ssize
    start = 0

    for i in range(categories):

        # settings = np.array(range(levels[i])) + 1
        coded = np.diag(np.matrix([1]*levels[i]).A1)
        nreps = ssize/ncycles
        ncycles = ncycles/levels[i]
        # settings = np.repeat(settings, nreps, axis = 0)
        coded = np.repeat(coded, nreps, axis = 1)
        # settings = np.reshape(settings, (-1,1))
        coded = coded.T
        # settings = np.repeat(settings, ncycles, axis = 1)
        # settings = np.reshape(settings.T, (-1,1))
        new_coded = coded
        for j in range(ncycles-1):
            new_coded = np.vstack((new_coded, coded))

        design = design.T
        design[start:start + levels[i],:] = new_coded.T
        start = start + levels[i]
        design = design.T

    return design

if __name__ == '__main__':
    modules = ['fuselage', 'wing', 'pod', 'VT', 'HT']
    namelist = {}
    namelist['fuselage'] = ['Cross section', 'awave']
    namelist['wing'] = ['delta', 'conventional', 'multi-section']
    namelist['pod'] = ['2 wing bot', '2 fuselage', '1 tail 2 fuselage', '1 tail 2 wing']
    namelist['VT'] = ['conventionalV']
    namelist['HT'] = ['conventionalH', 'canard', 'T-tail', 'tailess']
    levels = np.array([2, 3, 4, 1, 4], dtype = np.int)
    incompatible = (
        ('delta', 'conventionalH'),
        ('conventional', '2 wing bot'),
        ('tailess', 'conventional'),
        ('tailess', 'multi-section')
        )
    new_design = apply_compatibility(levels, modules, namelist, incompatible, self_compatible = True)
    m,n = new_design.shape
    print 'New number of cases = %d' % m
    print new_design

