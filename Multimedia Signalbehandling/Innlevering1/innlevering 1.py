import numpy as np
import datetime

P_Y = np.array([1/4, 1-1/4])
P_XgivenY = np.array([[1/2,1/2], # Rows: X=x, Columns Y=y
                        [1/2,1/2]])
P_X = P_XgivenY.dot(P_Y)

P_XandY = list()
P_XandY_log = list()
P_YgivenX = np.empty(shape=(2,2))
I_log = list()

for i in range(0,2):
    for j in range(0,2):
        P_XandY.append(P_X[i]*P_Y[j])
        P_XandY_log.append(P_X[j]*P_Y[i])
        temp = P_XgivenY[i,j]*P_Y[j]/P_X[i]
        P_YgivenX[i,j] = temp
        I_log.append(P_X[i]*P_Y[j]/(P_X[i]*P_Y[j]))
        
HX = -np.sum(P_X*np.log2(P_X))
HY = -np.sum(P_Y*np.log2(P_Y))
HXandY = -np.sum(np.array(P_XandY)*np.log2(np.array(P_XandY)))
HXgivenY = -np.sum(np.array(P_XandY)*np.log2(P_XgivenY.flatten()))
HYgivenX = -np.sum(np.array(P_XandY)*np.log2(P_YgivenX.flatten()))
IXY = HX - HXgivenY

print("H(X) = ",HX)
print("H(Y) = ",HY)
print("H(X,Y) = ",HXandY)
print("H(X|Y) = ",HXgivenY)
print("H(Y|X) = ",HYgivenX)
print("I(X;Y) = ",IXY)

def get_stationary_distribution(transition_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary_distribution = eigenvectors[:, np.isclose(eigenvalues, 1)].flatten()
    return stationary_distribution/np.sum(stationary_distribution)

def get_entropy_rate(transition_matrix):
    steady_state = get_stationary_distribution(transition_matrix)
    entropy_rate = 0
    for i in range(0, len(steady_state)):
        for j in range(0, len(steady_state)):
            entropy_rate += steady_state[i]*transition_matrix[i,j]*np.log2(transition_matrix[i,j]) if transition_matrix[i,j] != 0 else 0
    return -entropy_rate

transition_matrix = np.array([[7/8,1/8],[1/8,7/8]])
steady_state_distribution = get_stationary_distribution(transition_matrix)

print("entropy rate is ", get_entropy_rate(transition_matrix))

#######################################################################################################
data1 = np.loadtxt("data1.txt")
data1 = [str(int(i)) for i in data1]
data2 = np.loadtxt("data2.txt")
data2 = [str(int(i)) for i in data2]
data3 = np.loadtxt("data3.txt")
data3 = [str(int(i)) for i in data3]

datas = [data1, data2, data3]

def get_prob(data, max_val):
    symbols = dict()
    prob = list()
    for i in range(0, max_val+1):
        symbols[bin(i)[2:].zfill(len(data[0]))] = 0
    for i in data:
        symbols[i] += 1
    for key in symbols:
        prob.append(symbols[key]/len(data))
    return prob

P_datas = list()
for i in datas:
    prob = [i for i in get_prob(i,1) if i != 0] 
    P_datas.append(prob)

Hdatas = list()
for i in P_datas:
    Hdatas.append(-np.sum(i*np.log2(i)))

print("H(data1) = ",Hdatas[0])
print("H(data2) = ",Hdatas[1])
print("H(data3) = ",Hdatas[2])

##########################################################
def merge_bits(file, num):
    new_array = list()
    for i in range(0, len(file), num):
        try:
            word = ""
            for j in range(0, num):
                word += str(int(file[i+j]))
            new_array.append(word)
        except:
            pass
    return new_array

##########################################################
datas_2bit = list()
for i in datas:
    datas_2bit.append(merge_bits(i,2))

P_datas_2bit = list()
for i in datas_2bit:
    prob = [i for i in get_prob(i,3) if i != 0] 
    P_datas_2bit.append(prob)

Hdatas_2bit = list()
for i in P_datas_2bit:
    Hdatas_2bit.append(-np.sum(i*np.log2(i)))

print("H(data1_2bit) = ",Hdatas_2bit[0])
print("H(data2_2bit) = ",Hdatas_2bit[1])
print("H(data3_2bit) = ",Hdatas_2bit[2])
##########################################################
datas_3bit = list()
for i in datas:
    datas_3bit.append(merge_bits(i,3))

P_datas_3bit = list()
for i in datas_3bit:
    prob = [i for i in get_prob(i,7) if i != 0] 
    P_datas_3bit.append(prob)

Hdatas_3bit = list()
for i in P_datas_3bit:
    Hdatas_3bit.append(-np.sum(i*np.log2(i)))

print("H(data1_3bit) = ",Hdatas_3bit[0])
print("H(data2_3bit) = ",Hdatas_3bit[1])
print("H(data3_3bit) = ",Hdatas_3bit[2])
##########################################################
datas_4bit = list()
for i in datas:
    datas_4bit.append(merge_bits(i,4))

P_datas_4bit = list()
for i in datas_4bit:
    prob = [i for i in get_prob(i,15) if i != 0] 
    P_datas_4bit.append(prob)

Hdatas_4bit = list()
for i in P_datas_4bit:
    Hdatas_4bit.append(-np.sum(i*np.log2(i)))

print("H(data1_4bit) = ",Hdatas_4bit[0])
print("H(data2_4bit) = ",Hdatas_4bit[1])
print("H(data3_4bit) = ",Hdatas_4bit[2])
##########################################################
datas_10bit = list()
for i in datas:
    datas_10bit.append(merge_bits(i,10))

P_datas_10bit = list()
for i in datas_10bit:
    prob = [i for i in get_prob(i,1023) if i != 0] 
    P_datas_10bit.append(prob)

Hdatas_10bit = list()
for i in P_datas_10bit:
    Hdatas_10bit.append(-np.sum(i*np.log2(i)))

print("entropy rate of data1 = ",Hdatas_10bit[0]/10)
print("entropy rate of data2 = ",Hdatas_10bit[1]/10)
print("entropy rate of data3 = ",Hdatas_10bit[2]/10)
#when increasing n arbitrariliy the entroyp rate goes to zero, becuase the number of possible states increases exponentially,
#but most of the states will have a probability of zero and will not contribute to the entropy, then you divide by n and the entropy rate goes to zero.
##########################################################
#model datas_2bit as markov chain

def get_transition_matrix(data):
    symbols = dict()
    max_val = 2**len(data[0])
    for i in range(0, max_val):
        symbols[bin(i)[2:].zfill(len(data[0]))] = dict()
        for j in range(0, max_val):
            symbols[bin(i)[2:].zfill(len(data[0]))][bin(j)[2:].zfill(len(data[0]))] = 0
    for i in range(0, len(data)-1):
        symbols[data[i]][data[i+1]] += 1
    transition_matrix = np.zeros((max_val,max_val))
    row = 0
    for i in symbols:
        column = 0
        count = sum(symbols[i].values())
        for j in symbols[i]:
            transition_matrix[row,column] = symbols[i][j]/count if count != 0 else 0
            column += 1
        row += 1
    return transition_matrix     

datas_transition_matrix = [get_transition_matrix(i) for i in datas]

print("transition matrix of data1 = ",datas_transition_matrix[0])
print("transition matrix of data2 = ",datas_transition_matrix[1])
print("transition matrix of data3 = ",datas_transition_matrix[2])

print("steady state of data1 = ",get_stationary_distribution(datas_transition_matrix[0]))
print("steady state of data2 = ",get_stationary_distribution(datas_transition_matrix[1]))
print("steady state of data3 = ",get_stationary_distribution(datas_transition_matrix[2]))

print("entropy rate of data1 = ",get_entropy_rate(datas_transition_matrix[0]))
print("entropy rate of data2 = ",get_entropy_rate(datas_transition_matrix[1]))
print("entropy rate of data3 = ",get_entropy_rate(datas_transition_matrix[2])) 

###########################################################################################
# implement a run length encoder for data3 using N=7,15,31,63,127

def run_length_encoder(data, n):
    bit_length = int(np.log2(n+1))
    bol = False
    encoded = ""
    count = 0
    i = 0
    while i < len(data):
        if count==n:
            encoded += bin(count)[2:].zfill(bit_length)
            count = 0
        elif data[i] != str(int(bol)):
            encoded += bin(count)[2:].zfill(bit_length)
            count = 0
            bol = not bol
        else:
            count += 1
            i += 1
    encoded += bin(count)[2:].zfill(bit_length)
    if count==n:
        encoded += bin(0)[2:].zfill(bit_length)
    return encoded

def run_length_decoder(data, n):
    bit_length = int(np.log2(n+1))
    bol = False
    decoded = ""
    for i in range(0, len(data), bit_length):
        decimal = int(data[i:i+bit_length],2)
        decoded += str(int(bol))*decimal
        if decimal != n:
            bol = not bol
    return decoded

a = datetime.datetime.now()
encoded_data3_fast = run_length_encoder(datas[2], 63)

print("encoded data3 gain fast = ",len(encoded_data3_fast)/len(datas[2]))
print("encoded data3 size = ",len(encoded_data3_fast))

#using 7zip we get a file with size 1.56 Mb compared to our which is 1.84 Mb
    


