#
import os
import argparse
import subprocess
from TestFunc import * 
from os.path import exists


def imageList(dataPath):
    fileName=[]
    fileList=[]
#     if os.path.isfile(dataPath) and '.nii' in dataPath:
#         fileList+=[dataPath]
#         temp=dataPath
#         if '/' in temp: temp=temp.split('/')[-1]
#         fileName+=[temp.split('.nii')[0]]
#         print(fileName)

    if os.path.isdir(dataPath):
        
        fileList+=[d for d in os.listdir(dataPath) if '.nii.gz' in d]
        
        for temp in fileList:
            fileName+=[temp.split('.nii.gz')[0]]
         
    else:
        raise ValueError('Invalid data path input')
    
    fileList, fileName = zip(*sorted(zip(fileList, fileName)))
    return fileList, fileName

#skull strip
def skull_strip(inName, outName):
    subprocess.call(['bash', 'skull_strip.sh', inName, outName])
    fillHoles(outName)
    print(outName, 'skull strip done.')    


#run test 



if __name__== "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--modelPath', default='model_backup/epoch149_5class.pt')
    parser.add_argument('--outputPath', default='reconstructed')
    parser.add_argument('--dataPath', default='data-split/Scans')
    parser.add_argument('--betPath', default='data-split/skull-strip')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', default=200)
    parser.add_argument('--gtPath', default = 'data-split/gt')

    args = parser.parse_args()
    
    
    dataPath=args.dataPath
    modelPath=args.modelPath
    outputPath=args.outputPath
    betPath=args.betPath
    device=args.device
    BS=args.batch_size
    gtPath=args.gtPath

    fileList, fileName=imageList(dataPath)
    
    device=checkDevice(device)
    model=loadModel(modelPath, device)
    
    print('Total number:', len(fileName))
    for i in range(len(fileName)):
        
        if not exists(os.path.join(dataPath,'{}.nii.gz'.format(fileName[i]))):
            print(fileName[i], 'not exists')
            continue
      
        #skull_strip(os.path.join(dataPath, fileList[i]), os.path.join(betPath, fileName[i]))
        resultName=runTest(fileName[i], outputPath, dataPath, betPath, device, BS, model)
        
        
    
    
    
    
    
    
    
    
