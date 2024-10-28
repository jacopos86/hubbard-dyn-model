import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def read_file(filename,col_x,col_y):

    data= open (filename, 'r')
    datalines= data.readlines ()
    data.close ()

    len_file=len(datalines)

    com='#'
    j=-1
    x=[]
    y=[]

    while (com == '#'):
        j=j+1
        line= (datalines [j]).split ()
        com=line[0]
    while (len(line) != 0):
#        print line
        a=float (line [col_x - 1])
        b=float (line [col_y - 1])
#        print a,b
        x.append (a)
        y.append (b)
        j=j+1
        if (j == len_file):
            break
        line= (datalines [j]).split ()

    return np.array(x),np.array(y)

### plot occupations ###############
fig, ax = plt.subplots()

x,y1=read_file('GS2/Deltan.txt',1,2)
ax.plot(x,y1,linewidth=2,color='black',label=r'$\mathrm{U=0.0}$')

x,y2=read_file('GS3/Deltan.txt',1,2)
ax.plot(x,y2,linewidth=2,color='blue',label=r'$\mathrm{U=0.2}$')

x,y3=read_file('GS4/Deltan.txt',1,2)
ax.plot(x,y3,linewidth=2,color='red',label=r'$\mathrm{U=1.0}$')

x,y4=read_file('GS5/Deltan.txt',1,2)
ax.plot(x,y4,linewidth=2,color='violet',label=r'$\mathrm{U=2.0}$')

x,y5=read_file('GS6/Deltan.txt',1,2)
ax.plot(x,y5,linewidth=2,color='yellow',label=r'$\mathrm{U=5.0}$')

x,y6=read_file('GS7/Deltan.txt',1,2)
ax.plot(x,y6,linewidth=2,color='green',label=r'$\mathrm{U=10.0}$')

ax.legend(loc='best',handlelength=3,prop={'size':18},frameon=False)
#ax.set_xlim(-15.,15.)
#ax.set_ylim(0., 0.63)
ax.set_xlabel(r'$\mathrm{\Delta\,\,v}$',size=24)
ax.set_ylabel(r'$\mathrm{\Delta\,\,n}$',size=24)
mf = ticker.ScalarFormatter(useMathText=True)
mf.set_powerlimits((-1,1))
plt.gca().yaxis.set_major_formatter(mf)
#
plt.tick_params(axis='y', which='major', labelsize=15, width=2, length=5)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,-2))
plt.locator_params(axis='y', nbins=6)
plt.tick_params(axis='x', which='major', width=2, length=5, labelsize=0)
plt.minorticks_on()
plt.tick_params(axis='y', which='minor', width=1., length=2.5, labelsize=0)
plt.tick_params(axis='x', which='minor', width=1., length=2.5, labelsize=0)
#

plt.subplots_adjust(top=0.96)
plt.subplots_adjust(right=0.96)
plt.subplots_adjust(bottom=0.14)
plt.subplots_adjust(left=0.14)
plt.tick_params(labelsize=15)
plt.tick_params(labelsize=15)

plt.savefig ('dn.pdf', dpi=None, facecolor='w', edgecolor='w', format="pdf", transparent=False, bbox_inches=None, pad_inches=0.1)
plt.savefig ('dn.eps', dpi=None, facecolor='w', edgecolor='w', format="eps", transparent=False, bbox_inches=None, pad_inches=0.1)

plt.show()

####### plot energies ###############
fig, ax = plt.subplots()

x,y1=read_file('GS2/E.txt',1,2)
ax.plot(x,y1,linewidth=2,color='black',label=r'$\mathrm{U=0.0}$')

x,y2=read_file('GS3/E.txt',1,2)
ax.plot(x,y2,linewidth=2,color='blue',label=r'$\mathrm{U=0.2}$')

x,y3=read_file('GS4/E.txt',1,2)
ax.plot(x,y3,linewidth=2,color='red',label=r'$\mathrm{U=1.0}$')

x,y4=read_file('GS5/E.txt',1,2)
ax.plot(x,y4,linewidth=2,color='violet',label=r'$\mathrm{U=2.0}$')

x,y5=read_file('GS6/E.txt',1,2)
ax.plot(x,y5,linewidth=2,color='yellow',label=r'$\mathrm{U=5.0}$')

x,y6=read_file('GS7/E.txt',1,2)
ax.plot(x,y6,linewidth=2,color='green',label=r'$\mathrm{U=10.0}$')

ax.legend(loc='best',handlelength=3,prop={'size':18},frameon=False)
#ax.set_xlim(-15.,15.)
ax.set_ylim(-10., 0.1)
ax.set_xlabel(r'$\mathrm{\Delta\,\,v}$',size=24)
ax.set_ylabel(r'$\mathrm{E}$',size=24)
mf = ticker.ScalarFormatter(useMathText=True)
mf.set_powerlimits((-1,1))
plt.gca().yaxis.set_major_formatter(mf)
#
plt.tick_params(axis='y', which='major', labelsize=15, width=2, length=5)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,-2))
plt.locator_params(axis='y', nbins=6)
plt.tick_params(axis='x', which='major', width=2, length=5, labelsize=0)
plt.minorticks_on()
plt.tick_params(axis='y', which='minor', width=1., length=2.5, labelsize=0)
plt.tick_params(axis='x', which='minor', width=1., length=2.5, labelsize=0)
#
plt.subplots_adjust(top=0.96)
plt.subplots_adjust(right=0.96)
plt.subplots_adjust(bottom=0.14)
plt.subplots_adjust(left=0.16)
plt.tick_params(labelsize=15)
plt.tick_params(labelsize=15)

plt.savefig ('en.pdf', dpi=None, facecolor='w', edgecolor='w', format="pdf", transparent=False, bbox_inches=None, pad_inches=0.1)
plt.savefig ('en.eps', dpi=None, facecolor='w', edgecolor='w', format="eps", transparent=False, bbox_inches=None, pad_inches=0.1)

plt.show()
