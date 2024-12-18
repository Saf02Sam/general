#!/usr/bin/env python
# coding: utf-8

# In[5]:


str=input("str")
l=len(str)
def A(a,b):
    return a&b
def B(a,b):
    return a|b
def C(a,b):
    return a^b
if str[1]=='A':
    val=A(int(str[0]),int(str[2]))
if str[1]=='B':
    val=B(int(str[0]),int(str[2]))
if str[1]=='C':
    val=C(int(str[0]),int(str[2]))
for i in range(3,l-1,2):
    if str[i]=='A':
        val=A(val,int(str[i+1]))
    if str[i]=='B':
        val=B(val,int(str[i+1]))
    if str[i]=='C':
        val=C(val,int(str[i+1]))
print(val)


# In[29]:


st=input("str")
l=len(st)
a=st[0]
count=[0]*l
c=0
d=[a]
for i in range(0,l):
    if st[i]==a:
        count[c]=count[c]+1
    else:
        a=st[i]
        d.append(a)
        c=c+1
        count[c]=count[c]+1
for i in range(len(d)):
    print(d[i],count[i],sep='',end='')


# In[125]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target
x=df.drop(['target'],axis=1)
y=df['target']
knn=KNeighborsClassifier(n_neighbors=3)
xr,xs,yr,ys=train_test_split(x,y)
knn.fit(xr,yr)
k=knn.predict(xs)
acc=accuracy_score(ys,k)
print(acc)
x1=[5,3,1.4,0.2]
x1=pd.DataFrame(x1).T
y1=knn.predict(x1)
targets=iris.target_names
print(targets[y1])


# In[141]:


import numpy as np
from sklearn.cluster import KMeans
VAR=np.array([[1.713,1.586],[0.180,1.786],[0.353,1.240],[0.940,1.566],[1.486,0.759],[1.266,1.106],[1.540,0.419],[0.459,1.799],[0.773,0.186]])
CLASS=np.array([0,1,1,0,1,0,1,1,1])
km=KMeans(n_clusters=3)
km.fit(VAR,CLASS)
a=np.array([[0.906,0.606]])
k=km.predict(a)
print(k)


# In[147]:


import numpy as np
w=np.array([1,1])
b=-0.5
def OR(x):
    a=np.dot(w,x)+b
    if a>=0:
        print(x[0],x[1],1)
    else:
        print(x[0],x[1],0)
t1=np.array([0,0])
t2=np.array([0,1])
t3=np.array([1,0])
OR(np.array([1,1]))
OR(t1)
OR(t2)
OR(t3)
OR(t4)


# In[155]:


#sha1
import hashlib
str1=""
str2="abc"
str3="abcdefghijklmnopqrstuvwxyz"
res1=hashlib.sha1(str1.encode())
res2=hashlib.sha1(str2.encode())
res3=hashlib.sha1(str3.encode())
print("Algorithm = SHA1\nProvider = SUN version 1.6")
print('SHA1("")=',res1.hexdigest().upper())
print('SHA1("abc")=',res2.hexdigest().upper())
print('SHA1("abcdefghijklmnopqrstuvwxyz")=',res3.hexdigest().upper())


# In[157]:


#md5
import hashlib
str1=""
str2="abc"
str3="abcdefghijklmnopqrstuvwxyz"
res1=hashlib.md5(str1.encode())
res2=hashlib.md5(str2.encode())
res3=hashlib.md5(str3.encode())
print("Algorithm = SHA1\nProvider = SUN version 1.6")
print('MD5("")=',res1.hexdigest().upper())
print('MD5("abc")=',res2.hexdigest().upper())
print('MD5("abcdefghijklmnopqrstuvwxyz")=',res3.hexdigest().upper())


# In[164]:


#rsa
import math
import random
print("Enter 2 primes p and q")
p=int(input("p="))
q=int(input("q="))
n=p*q
phi_n=(p-1)*(q-1)
e=[]
for i in range(1,phi_n):
    if(math.gcd(i,phi_n)==1):
        e.append(i)
print("Values of e are:")
print(e)
random_e=random.choice(e)
print("Picked value of e is:")
print(random_e)
for i in range(1,phi_n):
    if(((random_e*i)%phi_n)==1):
        d=i
        break
print("Value of d is ",d)
m=int(input("Enter value for encryption"))
encrypt=(m**random_e)%n
decrypt=(encrypt**d)%n
print("Cipher text = ",encrypt)
print("Decrypted text = ",decrypt)


# In[431]:


#dhke
index=0
alpha=[None]*100
q=int(input("Enter the prime"))
for i in range(2,q):
    alpharnot=[None]*q
    for j in range(1,q):
        alpharnot[j-1]=(i**j)%q
    c=0
    for k in range(0,q):
        for p in range(k+1,q):
            if(alpharnot[k]==alpharnot[p]):
                c+=1
    if(c==0):
        alpha[index]=i
        index+=1
for i in range(0,index):
    print("Primitive root is ",alpha[i])
alpha_picked=int(input("Pick one of the root"))
xa=int(input("Pick XA"))
xb=int(input("Pick XB"))
ya=(alpha_picked**xa)%q
yb=(alpha_picked**xb)%q
ka=(yb**xa)%q
kb=(ya**xb)%q
print("KA = ",ka,"\nKB = ",kb)
if(ka==kb):
    print("Keys are same")


# In[432]:


#elgamal
index=0
invmod=0
alpha=[None]*100
q=int(input("Enter the prime"))
for i in range(2,q):
    alpharnot=[None]*q
    for j in range(1,q+1):
        alpharnot[j-1]=(i**j)%q
        c=0
        for k in range(0,q):
            for p in range(k+1,q):
                if(alpharnot[k]==alpharnot[p]):
                    c+=1
        if(c==0):
            alpha[index]=i
            index+=1
for i in range(0,index):
    print("Primitive root is ",alpha[i])
alpha_picked=int(input("Pick one of the root"))
xa=int(input("Pick XA"))
xb=int(input("Pick XB"))
ya=(alpha_picked**xa)%q
yb=(alpha_picked**xb)%q
print("Public key of A is : (",q,",",alpha_picked,",",yb,")")
print("Private key of A is : (",q,",",alpha_picked,",",xa,")")
print("Public key of B is : (",q,",",alpha_picked,",",ya,")")
print("Private key of B is : (",q,",",alpha_picked,",",xb,")")
print("Encryption")
k1=int(input("Pick the key"))
c1=(alpha_picked**k1)%q
plain=int(input("Enter plain text: "))
c2=(plain*(yb**k1))%q
print("C1=",c1,"\nC2=",c2)
print("Decryption")
for z in range(1,q):
    if((z*(c1**xb))%q==1):
        invmod=z
        break
plain1=(c2*invmod)%q
print("Decrypted text: ",plain1)


# In[1]:


#ceasers
ch=int(input("1.Shift 2.Shift-n"))
gen=""
if ch==1 or ch==2:
    pt=input("text")
    if ch==1:
        n=3
    else:
        n=int(input("n"))
    action=int(input("1.En 2.De"))
    if action==1:
        for c in pt:
            if(c.isupper()):
                gen+=chr(((ord(c))+n-65)%26+65)
            else:
                gen+=chr(((ord(c))+n-97)%26+97)
    elif action==2:
        for c in pt:
            if(c.isupper()):
                gen+=chr(((ord(c))-n-65)%26+65)
            else:
                gen+=chr(((ord(c))-n-97)%26+97)
print(gen)


# In[376]:


#hill
ch=int(input("1.En 2.De"))
if ch==1:
    key=[[],[]]
    print("Ent key")
    a=input("El 1,1:")
    key[0].append(ord(a)-97)
    a=input("El 1,2:")
    key[0].append(ord(a)-97)
    a=input("El 2,1:")
    key[1].append(ord(a)-97)
    a=input("El 2,2:")
    key[1].append(ord(a)-97)
    pt=[[None,None],[None,None]]
    p=input("Plain")
    pt[0][0]=ord(p[0])-97
    pt[0][1]=ord(p[1])-97
    pt[1][0]=ord(p[2])-97
    pt[1][1]=ord(p[3])-97
    ct=[[None,None],[None,None]]
    ct[0][0]=((key[0][0]*pt[0][0])+(key[0][1]*pt[0][1]))%26
    ct[0][1]=((key[1][0]*pt[0][0])+(key[1][1]*pt[0][1]))%26
    ct[1][0]=((key[0][0]*pt[1][0])+(key[0][1]*pt[1][1]))%26
    ct[1][1]=((key[1][0]*pt[1][0])+(key[1][1]*pt[1][1]))%26
    print("ct=")
    print(chr(ct[0][0]+97),chr(ct[0][1]+97),chr(ct[1][0]+97),chr(ct[1][1]+97))
elif ch==2:
    key=[[],[]]
    print("Ent key")
    a=input("El 1,1:")
    key[0].append(ord(a)-97)
    a=input("El 1,2:")
    key[0].append(ord(a)-97)
    a=input("El 2,1:")
    key[1].append(ord(a)-97)
    a=input("El 2,2:")
    key[1].append(ord(a)-97)
    pt=[[None,None],[None,None]]
    ct=[[None,None],[None,None]]
    c=input("Cipher")
    ct[0][0]=ord(c[0])-97
    ct[0][1]=ord(c[1])-97
    ct[1][0]=ord(c[2])-97
    ct[1][1]=ord(c[3])-97
    key1=[[None,None],[None,None]]
    k=(key[0][0]*key[1][1])-(key[0][1]*key[1][0])
    def invmod(a,b): return 0 if a==0 else 1 if b%a==0 else b - invmod(b%a,a)*b//a
    k=invmod(k,26)
    key1[0][0]=(((key[1][1])%26)*k)%26
    key1[0][1]=(((-key[0][1])%26)*k)%26
    key1[1][0]=(((-key[1][0])%26)*k)%26
    key1[1][1]=(((key[0][0])%26)*k)%26
    pt[0][0]=((key1[0][0]*ct[0][0])+(key1[0][1]*ct[0][1]))%26
    pt[0][1]=((key1[1][0]*ct[0][0])+(key1[1][1]*ct[0][1]))%26
    pt[1][0]=((key1[0][0]*ct[1][0])+(key1[0][1]*ct[1][1]))%26
    pt[1][1]=((key1[1][0]*ct[1][0])+(key1[1][1]*ct[1][1]))%26
    print("pt=")
    print(chr(pt[0][0]+97),chr(pt[0][1]+97),chr(pt[1][0]+97),chr(pt[1][1]+97))


# In[231]:


#railfence
import numpy as np

i, j, down = -1, 0, 0
text = input("Enter text : ")
n = int(input("Enter key : "))
ch = int(input("1. Encrypt\n2. Decrypt\nEnter your choice : "))
d = n
while len(text) >= d:
    d += 2 * (n - 1)
d = d - (2 * (n - 1))
text += "Q" * (len(text) - d)
c = text
i, j, k = -1, 0, 0
arr = np.full((n, len(text)), " ")
for j in range(len(text)):
    if i == (n - 1):
        down = -1
    if i == 0 or i == -1:
        down = 1
    i += down
    arr[i, j] = "*" if ch == 2 else text[j]
print(arr) if ch == 1 else print(end="")

if ch == 1:
    arr = arr.reshape(n * len(text))
    c = "".join([x + " " if x.isalnum() else " " for x in arr]).replace(" ", "")
    print(c)
else:
    for i in range(n):
        for j in range(len(text)):
            if arr[i, j] == "*":
                arr[i, j] = c[k]
                k += 1
        print("Rail ", i, " : ", arr[i])
    i, j, final = -1, 0, ""
    for j in range(len(text)):
        if i == (n - 1):
            down = -1
        if i == 0 or i == -1:
            down = 1
        i += down
        final += arr[i, j]

    print(final.replace("Q", ""))


# In[334]:


#railfence
import numpy as np
text = input("Enter text : ")
n = int(input("Enter key : "))
ch = int(input("1. Encrypt\n2. Decrypt\nEnter your choice : "))
d=n
while len(text) >= d:
    d += 2 * (n - 1)
d = d - (2 * (n - 1))
text += "Q" * (len(text) - d)
c=text
arr = np.full((n,len(text)), " ")
if ch==1:
    i,j,down,count=0,0,1,0
    arr[0][0]=text[0]
    for count in range(1,len(text)):
        if(down==1):
            i+=1
            j+=1
            arr[i][j]=text[count]
            if(i==n-1):
                down=-1
        elif(down==-1):
            i-=1
            j+=1
            arr[i][j]=text[count]
            if(i==0):
                down=1
    print(arr)
    for i in range(0,n):
        for j in range(0,len(text)):
            if arr[i][j].isalpha():
                print(arr[i][j],end='')
if ch==2:
    c=0
    for k in range(0,n):
        i,j,down=0,0,1
        if not(arr[i][j].isalpha()):
            arr[0][0]="-"
        for count in range(0,len(text)-1):
            if(down==1):
                i+=1
                j+=1
                if not(arr[i][j].isalpha()):
                    arr[i][j]="-"
                if(i==n-1):
                    down=-1
            elif(down==-1):
                i-=1
                j+=1
                if not(arr[i][j].isalpha()):
                    arr[i][j]="-"
                if(i==0):
                    down=1
        for l in range(0,len(text)):
            if arr[k][l]=='-':
                arr[k][l]=text[c]
                c+=1
        print(arr)
    for j in range(0,len(text)):
            for i in range(0,n):
                if arr[i][j].isalpha():
                    if arr[i][j]!='Q':
                        print(arr[i][j],end='')


# In[244]:


#railfence
import numpy as np
text = input("Enter text : ")
n = int(input("Enter key : "))
ch = int(input("1. Encrypt\n2. Decrypt\nEnter your choice : "))
while(((len(text)-n)%(2*(n-1)))!=0):
    text+='Q'
print(text)
arr = np.full((n,len(text)), " ")
if ch==1:
    i,j,down,count=0,0,1,0
    arr[0][0]=text[0]
    for count in range(1,len(text)):
        if(down==1):
            i+=1
            j+=1
            arr[i][j]=text[count]
            if(i==n-1):
                down=-1
        elif(down==-1):
            i-=1
            j+=1
            arr[i][j]=text[count]
            if(i==0):
                down=1
    print(arr)
    for i in range(0,n):
        for j in range(0,len(text)):
            if arr[i][j].isalpha():
                print(arr[i][j],end='')
if ch==2:
    c=0
    for k in range(0,n):
        i,j,down=0,0,1
        if not(arr[i][j].isalpha()):
            arr[0][0]="-"
        for count in range(1,len(text)):
            if(down==1):
                i+=1
                j+=1
                if not(arr[i][j].isalpha()):
                    arr[i][j]="-"
                if(i==n-1):
                    down=-1
            elif(down==-1):
                i-=1
                j+=1
                if not(arr[i][j].isalpha()):
                    arr[i][j]="-"
                if(i==0):
                    down=1
        for l in range(0,len(text)):
            if arr[k][l]=='-':
                arr[k][l]=text[c]
                c+=1
        print(arr)
    for j in range(0,len(text)):
            for i in range(0,n):
                if arr[i][j].isalpha():
                    if arr[i][j]!='Q':
                        print(arr[i][j],end='')


# In[337]:


#monoalpha
c=int(input("1.E 2.D"))
if c==1:
    t=input("Text")
    p=input("Perm").replace(" ","")
    a='abcdefghijklmnopqrstuvwxyz'
    for i in range(0,len(t)):
        for j in range(0,26):
            if t[i]==a[j]:
                print(p[j],end='')
if c==2:
    t=input("Text")
    p=input("Perm")
    a='abcdehghijklmnopqrstuvwxyz'
    for i in range(0,len(t)):
        for j in range(0,26):
            if t[i]==p[j]:
                print(a[j],end='')


# In[345]:


#one time pad
t=input("Text")
k=input("Key")
c=int(input("1.E 2.D"))
if c==1:
    if len(t)==len(k):
        for i in range(0,len(t)):
            print(chr((((ord(t[i])+ord(k[i]))-97)%26)+97),end='')
    else:
        print("Wrong key")
if c==2:
    if len(t)==len(k):
        for i in range(0,len(t)):
            print(chr((((ord(t[i])-ord(k[i]))-97)%26)+97),end='')
    else:
        print("Wrong key")


# In[352]:


#vigenere
t=input("Text")
k=input("Key")
l1=int(len(t)/len(k))
k=k*l1
for i in range(0,len(k)):
    if (len(k))!=len(t):
        k+=(k[i])
c=int(input("1.E 2.D"))
if c==1:
    for i in range(0,len(t)):
        print(chr((((ord(k[i])-97)+(ord(t[i])-97))%26)+97),end='')
if c==2:
    for i in range(0,len(t)):
        print(chr((((ord(t[i])-97)-(ord(k[i])-97))%26)+97),end='')


# In[419]:


#simple columnar
t=input("Text").replace(" ",'')
k=input("Key")
c=int(input("1.E 2.D"))
if c==1:
    while(len(t)%len(k)!=0):
        t+='Q'
    lt=len(t)
    lk=len(k)
    r=int(lt/lk)
    a=np.full((r+1,lk),'')
    for i in range(0,lk):
        a[0][i]=k[i]
    count=0
    for i in range(1,r+1):
        for j in range(0,lk):
            a[i][j]=t[count]
            count+=1
    print(a)
    k1=k
    for i in range(65,65+26):
        for j in range(0,lk):
            if chr(i)==k1[j]:
                for l in range(0,lk):
                    if k1[j]==a[0][l]:
                        for m in range(1,r+1):
                            print(a[m][l],end='')
                break
                k1.replace(k1[j],"",1)
if c==2:
    lk=len(k)
    lt=len(t)
    r=int(lt/lk)
    a=np.full((r+1,lk),'')
    a1=np.full((2,lk),'')
    for i in range(0,lk):
        a[0][i]=k[i]
        a1[0][i]=i
        a1[1][i]=k[i]
    a1=a1[:,a1[1,:].argsort()]
    a = a[:, a[0, :].argsort()]
    count=0
    for i in range(0,lk):
        for j in range(1,r+1):
            a[j][i]=t[count]
            count+=1
    a1=np.append(a1,a,axis=0)
    a1=a1[:,a1[0,:].argsort()]
    print(a1)
    for i in range(3,r+2):
        for j in range(0,lk):
            print(a1[i][j],end='')
    i=r+2
    for j in range(0,lk):
        if a1[i][j]!='Q':
            print(a1[i][j],end='')


# In[439]:


#simple columnar
t=input("Text").replace(" ",'')
k=input("Key")
c=int(input("1.E 2.D"))
if c==1:
    while(len(t)%len(k)!=0):
        t+='Q'
    lt=len(t)
    lk=len(k)
    r=int(lt/lk)
    a=np.full((r+1,lk),'')
    for i in range(0,lk):
        a[0][i]=k[i]
    count=0
    for i in range(1,r+1):
        for j in range(0,lk):
            a[i][j]=t[count]
            count+=1
    print(a)
    a=a[:,a[0,:].argsort()]
    for i in range(0,lk):
        for j in range(1,r+1):
            print(a[j][i],end='')
if c==2:
    lk=len(k)
    lt=len(t)
    r=int(lt/lk)
    a=np.full((r+1,lk),'')
    a1=np.full((2,lk),'')
    for i in range(0,lk):
        a[0][i]=k[i]
        a1[0][i]=i
        a1[1][i]=k[i]
    a1=a1[:,a1[1,:].argsort()]
    a = a[:, a[0, :].argsort()]
    count=0
    for i in range(0,lk):
        for j in range(1,r+1):
            a[j][i]=t[count]
            count+=1
    a1=np.append(a1,a,axis=0)
    a1=a1[:,a1[0,:].argsort()]
    print(a1)
    for i in range(3,r+2):
        for j in range(0,lk):
            print(a1[i][j],end='')
    i=r+2
    for j in range(0,lk):
        if a1[i][j]!='Q':
            print(a1[i][j],end='')


# In[429]:


c=[None]*20
d=[None]*20
a=input("\n\nEnter the input string : ")
l=len(a);
j=0
for i in range(0,l):
    if(i%2==0):
        c[j]=a[i]
        j+=1
for i in range(0,l):
    if(i%2==1):
        c[j]=a[i]
        j+=1
print("\nCipher text after applying rail fence :")
for i in c:
    if i!=None:
        print(i,end='')
if(l%2==0):
    k=int(l/2)
else:
    k=int((l/2)+1)
j=0
for i in range(0,k):
    d[j]=c[i]
    j=j+2
j=1
for i in range(0,k):
    d[j]=c[k+i]
    j=j+2
print("\nText after decryption : ")
for i in d:
    if i!=None:
        print(i,end='')


# In[3]:


print(3/2)


# In[2]:


pip install pycryptodomex


# In[17]:


from Cryptodome.Cipher import DES
def pad(text):
    n = len(text) % 32
    return text + (b'0' * (32-n))


key = b'-8B key-'
text1 = b'CRYPTOGRAPHY'

des = DES.new(key, DES.MODE_ECB)

padded_text = pad(text1)
encrypted_text = des.encrypt(padded_text)

print(type(encrypted_text))
print(encrypted_text)
s=des.decrypt(encrypted_text)
s=s.decode('ascii')
for i in s:
    if i!='0':
        print(i,end='')


# In[20]:


text1 = b'CRYPTOGRAPHY'
text1=text1.decode('UTF-8')
print(text1)


# In[3]:


from Cryptodome.Cipher import AES
def pad(text):
    n = len(text) % 32
    return text + (b'0' * (32-n))


key = b'1232132133211232'
text1 = b'CRYPTOGRAPHY'

des = AES.new(key, AES.MODE_ECB)

padded_text = pad(text1)
encrypted_text = des.encrypt(padded_text)
encrypted_text1 = encrypted_text.decode('utf-16')  

print(type(encrypted_text1))
print(encrypted_text1)
s=des.decrypt(encrypted_text)
s=s.decode('ascii')
for i in s:
    if i!='0':
        print(i,end='')


# In[2]:


from Cryptodome.Cipher import Blowfish
def pad(text):
    n = len(text) % 32
    return text + (b'0' * (32-n))


key = b'00002277'
text1 = b'CRYPTOGRAPHY'

des = Blowfish.new(key, Blowfish.MODE_ECB)

padded_text = pad(text1)
encrypted_text = des.encrypt(padded_text)
encrypted_text1 = encrypted_text.decode('utf-16')  

print(type(encrypted_text1))
print(encrypted_text1)
s=des.decrypt(encrypted_text)
s=s.decode('ascii')
for i in s:
    if i!='0':
        print(i,end='')


# In[ ]:


des
00aaaaaa
bf
00000011
00000033


# In[253]:


#railfence
import numpy as np
text = input("Enter text : ")
n = int(input("Enter key : "))
ch = int(input("1. Encrypt\n2. Decrypt\nEnter your choice : "))
while(((len(text)-n)%(2*(n-1)))!=0):
    text+='Q'
print(text)
arr = np.full((n,len(text)), " ")
if ch==1:
    i,j,down,count=0,0,1,0
    arr[0][0]=text[0]
    for count in range(1,len(text)):
        if(down==1):
            i+=1
            j+=1
            arr[i][j]=text[count]
            if(i==n-1):
                down=-1
        elif(down==-1):
            i-=1
            j+=1
            arr[i][j]=text[count]
            if(i==0):
                down=1
    print(arr)
    for i in range(0,n):
        for j in range(0,len(text)):
            if arr[i][j].isalpha():
                print(arr[i][j],end='')
if ch==2:
    c=0
    i,j,down=0,0,1
    arr[0][0]="-"
    for count in range(1,len(text)):
        if(down==1):
            i+=1
            j+=1
            arr[i][j]=text[c]
            c+=1
            if(i==n-1):
                down=-1
        elif(down==-1):
            i-=1
            j+=1
            arr[i][j]=text[c]
            c+=1
            if(i==0):
                down=1
        print(arr)
    for j in range(0,len(text)):
            for i in range(0,n):
                if arr[i][j].isalpha():
                    if arr[i][j]!='Q':
                        print(arr[i][j],end='')


# In[263]:


get_ipython().system('pip install des')


# In[1]:


import des
obj=des.new('abcdefgh', des.ECB)
plain="Guido van Rossum is a space alien."
len(plain)
obj.encrypt(plain)
ciph=obj.encrypt(plain+'XXXXXX')
ciph
obj.decrypt(ciph)


# In[9]:


from des import DesKey
key=DesKey(b'00aaaaaa')
key.encrypt(b'CRYPTOGRAPHY',padding=True).decode('utf-16')


# In[13]:


get_ipython().system('pip install aes_cipher')


# In[15]:


from aes_cipher import DataEncrypter
data_encrypter = DataEncrypter(Pbkdf2Sha512Default)
data_encrypter.Encrypt(data, "test_pwd")
enc_data = data_encrypter.GetEncryptedData()


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
a=np.full((35,2),0)


# In[9]:


from Cryptodome.Cipher import DES
def pad(text):
    n = len(text) % 32
    return text + (b'0' * (32-n))

try:
    # take path of image as a input
    path = input(r'Enter path of Image : ')
     
    # taking encryption key as input
    key1 = b'a'
    key2 = b'b'
    key3 = b'c'
     
    # print path of image file and encryption key that
    # we are using
     
    # open file for reading purpose
    fin = open(path, 'rb')
     
    # storing image data in variable "image"
    image = fin.read()
    fin.close()
     
    # converting image into byte array to 
    # perform encryption easily on numeric data
    image = bytearray(image)
 
    # performing XOR operation on each value of bytearray
    for index, values in enumerate(image):
        image[index] = tripdesenc(values,key1)
    for index, values in enumerate(image):
        image[index] = tripdesdec(values,key2)
    for index, values in enumerate(image):
        image[index] = tripdesenc(values,key3)
 
    # opening file for writing purpose
    fin = open(path, 'wb')
     
    # writing encrypted data in image
    fin.write(image)
    fin.close()
    print('Encryption Done...')
 
     
except Exception:
    print('Error caught : ', Exception.__name__)

def tripdesenc(values,key):
    des = DES.new(key, DES.MODE_ECB)

    padded_text = pad(values)
    encrypted_text = des.encrypt(padded_text)
    return encrypted_text
def tripdesdec(values,key):
    s=des.decrypt(encrypted_text)
    return s

'''print(type(encrypted_text))
print(encrypted_text)
s=des.decrypt(encrypted_text)
s=s.decode('ascii')
for i in s:
    if i!='0':
        print(i,end='')'''


# In[ ]:


b'-8B key-'
"C:\Users\Saffa\OneDrive\Pictures\Screenshots\Screenshot_20230105_003129.png"
"http://localhost:8888/tree/Screenshot_20230105_000643.png"


# In[ ]:


from Cryptodome.Cipher import DES
def pad(text):
    n = len(text) % 32
    return text + (b'0' * (32-n))

try:
    # take path of image as a input
    path = input(r'Enter path of Image : ')
     
    # taking encryption key as input
    key1 = b'a'
    key2 = b'b'
    key3 = b'c'
     
    # print path of image file and encryption key that
    # we are using
    print('The path of file : ', path)
    print('Key for encryption : ', key)
     
    # open file for reading purpose
    fin = open(path, 'rb')
     
    # storing image data in variable "image"
    image = fin.read()
    fin.close()
     
    # converting image into byte array to 
    # perform encryption easily on numeric data
    image = bytearray(image)
 
    # performing XOR operation on each value of bytearray
    for index, values in enumerate(image):
        image[index] = tripdesdec(values,key3)
    for index, values in enumerate(image):
        image[index] = tripdesenc(values,key2)
    for index, values in enumerate(image):
        image[index] = tripdesdec(values,key1)
 
    # opening file for writing purpose
    fin = open(path, 'wb')
     
    # writing encrypted data in image
    fin.write(image)
    fin.close()
    print('Decryption Done...')
 
     
except Exception:
    print('Error caught : ', Exception.__name__)

def tripdesenc(values,key):
    des = DES.new(key, DES.MODE_ECB)

    padded_text = pad(text1)
    encrypted_text = des.encrypt(padded_text)
    return encrypted_text

def tripdesdec(values,key):
    s=des.decrypt(encrypted_text)
    return s

