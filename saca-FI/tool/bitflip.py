from struct import *
def bitflip0(x, pos):
    fs = pack('f', x)
    bval = list(unpack('BBBB', fs))
    [q, r] = divmod(pos, 8)
    bval[q - 1] ^= 1 << r
    fs = pack('BBBB', *bval)
    fnew = unpack('f', fs)
    return fnew[0]

def bitflip(x, pos):
    flag=0
    fs = pack('f', x)
    bval = list(unpack('BBBB', fs))
    [q, r] = divmod(pos, 8)
    origin=bval[q]
    bval[q] ^= 1 << r
    if bval[q]>origin:
        flag=1
    fs = pack('BBBB', *bval)
    fnew = unpack('f', fs)
    return fnew[0],flag

def bitflip1(x, pos,fix):
    flag=0
    fs = pack('f', x)
    bval = list(unpack('BBBB', fs))
    [q, r] = divmod(pos, 8)
    origin=bval[q]
    bval[q] ^= 1 << r
    if bval[q]>origin:
        flag=1
    fs = pack('BBBB', *bval)
    fnew = unpack('f', fs)
    if fix==1:
        if flag==1:
            return fnew[0]
        else:
            return x
    if fix==0:
        if flag==1:
            return x
        else:
            return fnew[0]

def bitflip2(x, pos):
    Flag_list=[]
    if type(pos)==type(Flag_list):
        for i in pos:
            fnew,sign=bitflip(x,i)
            Flag_list.append(sign)
            x=fnew
        return fnew,Flag_list
    else:
        fnew, flag=bitflip(x,pos)
        return fnew, flag