import train_SdA12
import test_SdA_circ


train_SdA12.main('E:/personal/KFU/rsda.60-48-40-32/model/rsda1.dat',
                 'E:/personal/KFU/train/train601.csv',
                 'E:/personal/KFU/rsda.60-48-40-32/model/layers1.dat',
                 ins=60, layers_sizes=[48, 40, 32], recurrent_layer = 2)
test_SdA_circ.main('E:/personal/KFU/test601',
                   'E:/personal/KFU/rsda.60-48-40-32/encoded1',
                   'E:/personal/KFU/rsda.60-48-40-32/model/rsda1.dat')
