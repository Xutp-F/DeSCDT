# ---------------------------------
# --------人生苦短，我用python--------
# ---------------------------------
# ---------------------------------
# --------人生苦短，我用python--------
# ---------------------------------
import preprocess as pp
import re
import os
 # 后期在这中添加一个测试程序，先保证编译成功再产生数据

def generate_training_data(text):
    maxlen = 50
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen - 1):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
        if((len(sentences[i])==50) and (len(next_chars[i])==1)):
            data = sentences[i] + "/-*/"+ next_chars[i]+"\n"
            open("data.txt", "a").write(data)
        else:
            print(sentences[i])


path = './v0_4_code'
files = []
valid_count = 0
errorpath = []
datatrain = []
for root, d_names, f_names in os.walk(path):
    for f in f_names:
        files.append(os.path.join(root, f))

for file in files:
    if(file in errorpath):
        continue
    text = open(file, 'r').read()
    generate_training_data(text)
