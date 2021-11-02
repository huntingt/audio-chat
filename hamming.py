def nib_to_bits(nib):
    return [
        (nib >> 0) & 1,
        (nib >> 1) & 1,
        (nib >> 2) & 1,
        (nib >> 3) & 1
    ]

def bits_to_nib(bits):
    return bits[0] | (bits[1]<<1) | (bits[2]<<2) | (bits[3]<<3)

def encode_nibble(nib):
    d = nib_to_bits(nib)
    p0 = d[0] ^ d[1] ^ d[3]
    p1 = d[0] ^ d[2] ^ d[3]
    p2 = d[1] ^ d[2] ^ d[3]
    return [p0, p1, d[0], p2, d[1], d[2], d[3]]

def encode(nibs):
    bits = []
    for nib in nibs:
        bits += encode_nibble(nib)

    # round message up for zip
    bits += [0, 0, 0]
    
    encoded = []
    for i in range(0, len(bits) - 3, 4):
        encoded.append(bits_to_nib(bits[i:][:4]))

    return encoded

def syndrom(b):
    z1 = b[0] ^ b[2] ^ b[4] ^ b[6]
    z2 = b[2] ^ b[3] ^ b[5] ^ b[6]
    z3 = b[3] ^ b[4] ^ b[5] ^ b[6]
    z = z1 | (z2<<1) | (z3<<2)
    if z > 0:
        b[z - 1] ^= 1
    return bits_to_nib((b[2], b[4], b[5], b[6]))
        
def decode(nibs):
    bits = []
    for nib in nibs:
        bits += nib_to_bits(nib)

    decoded = []
    for i in range(0, len(bits) - 6, 7):
        decoded.append(syndrom(bits[i:][:7]))

    return decoded

if __name__ == "__main__":
    x = list(range(2))
    enc = encode(x)
    enc[0] += 1
    dec = decode(enc)
    print(f"{x} -> {enc} -> {dec}")
