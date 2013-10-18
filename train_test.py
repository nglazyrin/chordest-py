import train_SdA12
import test_SdA_circ
import string
import os
from base_train_SdA import MyException

def doTrainTest(layers, recurrent_layer, folderSuffix = ''):
    prefix = os.path.join('E:/personal', 'dissertation')
    model = 'sda' if recurrent_layer < 0 else 'rsda'
    targetFolder = os.path.join(prefix, model + '.' +
        string.join([str(l) for l in layers], '-') + folderSuffix)
    if not os.path.exists(targetFolder): os.makedirs(targetFolder)
    print targetFolder
    for i in [0, 1]:
        suffix = str(i) if layers[0] == 48 else '60' + str(i)
        train = os.path.join(prefix, 'data', 'train', 'train' + suffix + folderSuffix + '.csv')
        test = os.path.join(prefix, 'data', 'test' + suffix + folderSuffix)
        modelFolder = os.path.join(targetFolder, 'model')
        if not os.path.exists(modelFolder): os.makedirs(modelFolder)
        modelFile = os.path.join(modelFolder, model + str(i) + '.dat')
        layersFile = os.path.join(modelFolder, 'layers' + str(i) + '.dat')
        continue_looping = True
        print 'Train file: ' + train
        print 'Test dir: ' + test
        print 'Model file: ' + modelFile
        print 'Layers file: ' + layersFile
        while continue_looping and not os.path.exists(modelFile):
            try:
                train_SdA12.main(model_file = modelFile,
                         train_file = train,
                         layers_file = layersFile,
                         ins=layers[0], layers_sizes=layers[1:],
                         recurrent_layer = recurrent_layer)
                continue_looping = False
            except MyException:
                print 'Restarting training because of NaN'
                continue_looping = True
        encodedFolder = os.path.join(targetFolder, 'encoded' + str(i))
        if not os.path.exists(encodedFolder): os.makedirs(encodedFolder)
        test_SdA_circ.main(test, encodedFolder, modelFile)
    print '-------------------------------------------------------------------'
    
def main():
#    doTrainTest([48, 36], 0)
#    doTrainTest([48, 36], -1)
#    doTrainTest([48, 96, 48], -1)
#    doTrainTest([48, 96, 48], 1)
#    doTrainTest([48, 40, 32], -1)
#    doTrainTest([48, 40, 32], 1)
#    doTrainTest([48, 96, 48], -1, '.nolog')
#    doTrainTest([48, 96, 48], 1, '.nolog')
    doTrainTest([60, 150, 100], -1)
    doTrainTest([60, 150, 100], 1)

if __name__ == '__main__':
    main()
