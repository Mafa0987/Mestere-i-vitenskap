import numpy as np

def run_length_encoder(data, n):
    bit_length = int(np.log2(n+1))
    bol = False
    bits = {False: "0", True: "1"}
    encoded = ""
    count = 0
    for i in range(0, len(data)):
        if data[i] != bits[bol]:
            if count < n:
                encoded += bin(count)[2:].zfill(bit_length)
            else: 
                encoded += bin(n)[2:].zfill(bit_length)
                count -= n
                while count >= 0:
                    if count < n:
                        encoded += bin(count)[2:].zfill(bit_length)
                    else:
                        encoded += bin(n)[2:].zfill(bit_length)
                    count -= n
            count = 1
            bol = not bol
        else:
            count += 1
    if count < n:
        encoded += bin(count)[2:].zfill(bit_length)
    else: 
        encoded += bin(n)[2:].zfill(bit_length)
        count -= n
        while count >= 0:
            if count < n:
                encoded += bin(count)[2:].zfill(bit_length)
            else:
                encoded += bin(n)[2:].zfill(bit_length)
            count -= n
    return encoded

def run_length_encoder_fast(data, n):
    bit_length = int(np.log2(n+1))
    bol = False
    bits = {False: "0", True: "1"}
    encoded = ""
    count = 0
    i = 0
    while i < len(data):
        if count==n:
            encoded += bin(count)[2:].zfill(bit_length)
            if data[i] != bits[bol]:
                encoded += bin(0)[2:].zfill(bit_length)
            count = 0
        elif data[i] != bits[bol]:
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

s = "111111111111110"
print(run_length_encoder([*s], 7))
print(run_length_encoder_fast([*s], 7))