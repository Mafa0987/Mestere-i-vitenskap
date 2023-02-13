import numpy as np

def run_length_encoder_noob(data, n):
    bit_length = int(np.log2(n+1))
    bol = False
    encoded = ""
    count = 0
    i = 0
    while i < len(data):
        if count==n:
            encoded += bin(count)[2:].zfill(bit_length)
            count = 0
            bol = not bol
        elif data[i] != str(int(bol)):
            encoded += bin(count)[2:].zfill(bit_length)
            count = 0
            bol = not bol
        else:
            count += 1
            i += 1
    encoded += bin(count)[2:].zfill(bit_length)
    return encoded

s = "1111111111111101"

def run_length_decoder(data, n):
    bit_length = int(np.log2(n+1))
    bol = False
    decoded = ""
    for i in range(0, len(data), bit_length):
        decimal = int(data[i:i+bit_length],2)
        decoded += str(int(bol))*decimal
        bol = not bol
    return decoded

print(run_length_encoder_noob(s, 7))
print(run_length_decoder(run_length_encoder_noob(s, 7),7))