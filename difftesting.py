import os
import shutil
import subprocess
import obtain_Inputs

PROJECT_DIR = os.getcwd()   # NEED TO BE MODIFIED
contractPATH = PROJECT_DIR + "/contract/"
outPATH = PROJECT_DIR + "/Out/"
need_prefix = True
ori = "python3 " + PROJECT_DIR + "/myEVM/runTx.py --code A --data B"

def create_folder(dirPATH):
    """
    Create folders needed
    """
    binPATH = dirPATH + "bincode/"
    seedPATH = dirPATH + "seed/"
    outputPATH = dirPATH + "output/"

    if os.path.exists(binPATH) == False :
        os.makedirs(binPATH)
    if os.path.exists(seedPATH) == False :
        os.makedirs(seedPATH)
    if os.path.exists(outputPATH) == False :
        os.makedirs(outputPATH)
    return binPATH, seedPATH, outputPATH


def main():
    # args = parse_args()
    global curC
    curC = 0
    index = 0
    if os.path.exists(outPATH):
        shutil.rmtree(outPATH)
    os.makedirs(outPATH)


    for _, _, filenames in os.walk(contractPATH, followlinks=True):
        # Step 1.遍历合约
        for fileName in filenames:
            curC += 1

            dirPATH = outPATH + "contract" + str(curC) + "/"
            binPATH, seedPATH, outputPATH = create_folder(dirPATH)

            retcode = subprocess.call("solc " + contractPATH + fileName + " --hashes -o " + dirPATH, shell=True)
            if (retcode == 1):
                os.remove(contractPATH + fileName)
                continue

            for f in os.listdir(dirPATH):
                if os.path.splitext(f)[1] == '.signatures':
                    sigfile = open(dirPATH + f, "r")
                    line = sigfile.read().splitlines()
                    for signature in line:
                        pos1 = signature.find('(')
                        pos2 = signature.find(')')
                        funcName = signature[10:pos1]

                        sig = "0x" + signature[:8]
                        dataList = signature[pos1+1:pos2].split(',')
                        # check parameter type
                        canDeal = True
                        for i, val in enumerate(dataList) :
                            if (val.find('bool') == -1) and (val.find('uint') == -1) and (val.find('int') == -1) and (val.find('address') == -1 and (val.find("") == -1)):
                                canDeal = False
                                break
                            if (val.find('[') != -1) and (val.find(']') != -1) : # 数组
                                canDeal = False
                                break
                        if canDeal == False :
                            continue
                        inputData = obtain_Inputs.make(dataList)
                        sigName = sig + inputData

                        retcode = subprocess.call(
                            "solc --bin-runtime " + contractPATH + fileName + " -o " + binPATH + "bincode_" + funcName + "_" + str(index), shell=True)
                        # retcodeoptimize = subprocess.call(
                        #     "solcjs --bin  " + contractPATH + fileName + " -o " + binPATH + "bincode_" + funcName + "_" + "optimize" + str(
                        #         index), shell=True)
                        retcodeoptimize = subprocess.call(
                            "solc --bin-runtime --optimize " + contractPATH+fileName  + " -o " + binPATH + "bincode_" + funcName + "_"+"optimize" + str(
                                index), shell=True)

                        path_list = binPATH + "bincode_" + funcName + "_" + str(index)
                        for f in os.listdir(path_list):
                            if os.path.splitext(f)[1] == ".bin-runtime":
                                path_list_1 = path_list + "/" + f
                        path_list_optimize = binPATH + "bincode_" + funcName + "_"+"optimize" + str(index)
                        for f in os.listdir(path_list_optimize):
                            if os.path.splitext(f)[1] == '.bin-runtime':
                                path_list_optimize_1 = path_list_optimize + "/" + f
                        # liangge shixian zidong path
                        codefile = open(path_list_1, "r")

                        codefileOptimize = open(path_list_optimize_1, "r")
                        bincode = codefile.read()
                        bincodeOptimize = codefileOptimize.read()

                        tmp = ori.replace("A", bincode)
                        if need_prefix:
                            cmd = tmp.replace("B", sigName)
                        else:
                            cmd = tmp.replace("B", sigName[2:])

                        tmpOptimize = ori.replace("A", bincodeOptimize)
                        if need_prefix:
                            cmdOptimize = tmpOptimize.replace("B", sigName)
                        else:
                            cmdOptimize = tmpOptimize.replace("B", sigName[2:])
                        # print(cmd)
                        retcode = subprocess.call(cmd + " > " + outputPATH + str(funcName) + "myout.json",
                                                  shell=True)
                        retcode = subprocess.call(cmdOptimize + " > " + outputPATH + str(funcName)+ "myoutOptimize.json",
                                                  shell=True)


if __name__ == "__main__":
    main()