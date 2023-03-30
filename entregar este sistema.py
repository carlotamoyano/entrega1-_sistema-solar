import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib import pyplot as plt

G=6.67*10**(-15) #N*km**2/kg**2
c=1.496*10**8


#distancia (x,y) del sol a los planetas (km)
ro=np.array([(0,0),(57909227,0),(108.21*10**6,0),(149597870.7,0),(227940000,0),(778330000,0),(1429400000,0),(2870990000,0),(4504300000,0)])
#velocidad orbital media planetas (vx,vy) (km/h)
#vo=np.array([(0,0),(0,170.503),(0,126.07704),(0,107.208),(0,86.760),(0,47.160),(0,34.560),(0,24.480),(0,19.440)])
mo=np.array([1.989*10**30,3.303*10**23,4.869*10**24,5.976*10**24,6.421*10**23,1.9*10**27,5.69*10**26,8.6810*10**25,1.0241*10**26 ])


#cambios de variables   
rc=np.divide(ro,c)
#t1=((G*mo[0]/c**3)**(1/2))*t==58.1 días................................................................................................7
mc=np.divide(mo,mo[0])
#vo=vo*(1/c)*10**2*(G*mo[0]/c**3)**(-0.5)   hice la conversión a mano pq me daban mal:
vo=np.array([((0,0),(0,1.591251594),(0,1.174974805),(0,1.00040712),(0,0.8090540803),(0,0.4397762843),(0,0.3256358746),(0,0.2282808193),(0,0.18128818271))])

    
#trasponemos para tener las matrices ordenadas por filas y no columnas                                                                                               
rc=np.transpose(rc)
vo=np.transpose(vo)
print(rc)
print(vo)

t=0
h=0.1
#DEFINO UNA FUNCIÓN PARA LA ACELERACIÓN 
#a=masa, b=rc 
def ach(a,b):
    x=np.zeros(len(a))
    y=np.zeros(len(a))
    #matriz aceleraciones
    A=np.array([x,y])
    for i in range(len(a)):
        for k in range(2):
            matriz=0
            for j in range(len(a)):
                Akj=0
                if j!= i:
                    Akj=-a[j]*(b[k][i]-b[k][j])/(np.sqrt((b[0][i]-b[0][j])**2+(b[1][i]-b[1][j])**2))**3
                    matriz=matriz+Akj
                else:#no se afecta un planeta a sí mismo
                        matriz=matriz+0
                        A[k][i]=matriz
    return A

#DEFINO UNA FUNCIÓN PARA LA W
#d=velocidad e=aceleración f=m 
def velang(d,e,f,h):
    x=np.zeros(len(mo))
    y=np.zeros(len(mo))
    #matriz velocidades angulares
    W=np.array([x,y])
    for i in range(len(mo)):
        for k in range(2):
            W[k][i]=d[k][i]+(h/2)*e[k][i]
    return W
    
#DEFINO UNA FUNCIÓN PARA LA POSICIÓN
#r=ro v=vo  a=A
def pos(r,w,h):
    x=np.zeros(len(mo))
    y=np.zeros(len(mo))
    #matriz posiciones en tiempo h
    R=np.array([x,y])
    for i in range(len(mo)):
        for k in range(2):
            #R[k][i]=r[k][i]+h*v[k][i]+((h**2)/2)*a[k][i]
            R[k][i]=r[k][i]+h*w[k][i]
    return R    

#DEFINO UNA FUNCIÓN PARA LA VELOCIDAD
#w= ;a=ach
def vel(w,a,h):
    x=np.zeros(len(mo))
    y=np.zeros(len(mo))
    #matriz velocidades
    V=np.array([x,y])
    for i in range(len(mo)):
        for k in range(2):
            V[k][i]=w[k,i]+(h/2)*a[k][i]
    return V
#calculamos la energía de cada órbita
def Energia(mc,vo,rc,G):
    E=0
    K=0
    U=0
    for i in range(len(mo)):
        K = K+(1/2) * mc[i] * (vo[1][i]**2+vo[0][i]**2)
    
        for j in range(len(mo)):
            if j!=i:
                U = U - G * mc[i] * mc[j] / (((rc[0][i]-rc[0][j])**2 +(rc[1][i])-rc[1][j])**2)**0.5
                E = K+ U
    return E


T=np.zeros(len(mo))

def Periodo(rc,t):   
    for i in range(len(mo)):
        if rc[1][i]<0 and T[i]==0:
            T[i]=t
    return T

f=open("posiciones1(t).txt","w")


while t<1000: # calcula las posiciones, velocidades y aceleraciones, si lo hiciera durante una año terrestre tendría que ser t=t1, pero así no tendríamos todos los períodos

#guardamos los datos obtenidos para cada t en un fichero

    with open("velocidades(t).txt", "a") as file:        
        for v in vo:
            file.write(str(vo)+"\n")    
#comprobamos que esté funcionando correctamente         
   # print("rc=",rc) 
    #print("vo=",vo)
    
    posiciones=np.transpose(rc)
    np.savetxt(f,posiciones,newline="\n", delimiter=",")  
    f.write("\n")
#situándolo nada más comenzar la función pierdo el último valor que me calcula mi función en el fichero
#pero si no perdería los valores iniciales.

#calculo las aceleraciones a tiempo t           
    aceleracion=ach(mc,rc)
#guardamos las aceleraciones en un fichero
    with open("aceleraciones(t).txt", "a") as file:
        for a in aceleracion:
            file.write(str(aceleracion) + "\n")

#calculamos la velocidad angular a tiempo t
    velangular=velang(vo,aceleracion,mc,h)


#calculamos la siguiente posición r(t+h)
    rc=pos(rc,velangular,h) 


#aceleración a tiempo t+h
#se me va a sobreescribir la anterior aceleración para usarla en el siguiente t+h
    aceleracion=ach(mc,rc)  

    
#calculamos la velocidad a tiempo t+h
    vo=vel(velangular,aceleracion,h)
#como se va a sobreescribir con las velocidades iniciales del bucle debo poner q se guarden en un fichero para cada t al principio del mismo

#calcular el periodo
    Per=2*Periodo(rc,t)
    print("Periodo=",Per)
#comprobar que la energía es cte
    t2= np.arange(0, 1000, 0.1) #pq realmente nunca llega al 1000, ya que sale del bucle antes, así que ese último valor caería a cero al no guardarse
    E=np.zeros(len(t2))
    E1=Energia(mc,vo,rc,G)
    E[0]=0.0001110487736356882
    print("E=",E1) #para ir comprobando que da cte, antes de hacer el plot frente a t
    for i in range(len(t2)-1):
        if E[i]==0 and E[i-1]!=0 and E[i+1]==0:
            E[i]=E1
    
    
#el paso del tiempo
    t=t+h
    
    
f.close()


#OBTUVIMOS E=cte (variaba muy muy poco, siempre rondaba el valor 0.000111)
#Nuestros períodos son: [   0.     1.4    3.8    6.2   11.6   75.2  189.   527.  1019.4], cero para al sol pq lo consideramos estático y T=365/t1=365/58.1=6.282 para la tierra
#Un año en Urano es 84 años en la tierra, 84*6.282=527.69, básicamente nuestro resultado.
#En neptuno son 165 años terrestres, 165*6.282=1036.53
#Saturno: 29.5*6.282=185.32
#Júpiter: 10.8*6.282=67.85
#Marte: (687/365)*6.282=11.8
#Venus: (225/365)*6.282=3.87
#Mercurio: (88/365)*6.282=1.51
#Bastante similares

plt.plot(t2,E)

# Calculamos el momento angular de cada planeta
#l= np.zeros(len(mo))
#for i in range(len(mo)):
   # l[i]=mo[i]*((np.linalg.norm(rc[:,i])**3 *((np.linalg.norm(aceleracion[:,i]))+G*mo[0]*((np.linalg.norm(rc[:,i])))


#COMPROBAR QUE LAS TRAYECTORIAS SON ELIPSES
#eps= np.zeros(len(mo))
#for i in range(len(mo)):
   # eps[i]=((1+2*E[i]*l[i]**2)/(G**2)*(mo[0]**2)*mo[i]**3)**0.5
    #if eps[i]<1:
       # print("La trayectoria es una elipse")
