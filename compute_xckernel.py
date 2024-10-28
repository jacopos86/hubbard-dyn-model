import numpy as np
import os
import sys
from Set_susceptibility import Susceptibility

##################################################
######### XC KERNEL CALCULATION ##################
##################################################

###### EXTRACT DATA FROM INPUT FILE ##############

f = open("input3", "r")
lines = f.readlines()
outdir = "XCK4/"
os.mkdir(outdir)

##### TWO SITES EXTERNAL SCALAR POTENTIAL ########
vext = np.zeros(2)

for line in lines:
	l = line.strip()
	l = l.split()
	if len(l) > 0:
		### RUN TYPE ######
		if l[0] == "run":
			run = l[2]
		### HOPPING #######
		if l[0] == "t":
			t = float(l[2])
		### U #############
		if l[0] == "U":
			U = float(l[2])
		### maxiter #######
		if l[0] == "maxstep":
			maxstep = int(l[2])
		### beta ##########
		if l[0] == "beta":
			beta = float(l[2])
		### toler #########
		if l[0] == "tol":
			tol = float(l[2])
		### ext. pot. #####
		if l[0] == "vext":
			vext[0] = float(l[2])
			vext[1] = float(l[4])
		### time T ########
		if l[0] == "T":
			T = float(l[2])
		### dt ############
		if l[0] == "dt":
			dt = float(l[2])
	else:
		pass
		
##### COMPUTE HUBBARD SUSCEPTIBILITY #############

chiH = Susceptibility( T, dt )
chiH.set_chi_oft("Hubbard", U, t, vext, maxstep, tol, beta)

##### COMPUTE KS SUSCEPTIBILITY ##################

chiKS = Susceptibility( T, dt )
chiKS.set_chi_oft("Kohn-Sham", U, t, vext, maxstep, tol, beta)

##### COMPUTE XC KERNEL IN TIME ##################
##### fHxc = \chiKS^-1 - chi^-1 ##################
fHxc = np.zeros((2,2,chiH.Nt), dtype=np.complex128)
for i1 in range(2):
	for i2 in range(2):
		fHxc[i1,i2,:] = chiH.chi_oft[i1,i2,:] - chiKS.chi_oft[i1,i2,:]
		for t in range(chiH.Nt):
			if abs(fHxc[i1,i2,t]) < 1.e-7:
				fHxc[i1,i2,t] = 0.
			if abs(chiH.chi_oft[i1,i2,t]) < 1.e-15:
				chiH.chi_oft[i1,i2,t] = 1.e-10
			if abs(chiKS.chi_oft[i1,i2,t]) < 1.e-15:
				chiKS.chi_oft[i1,i2,t] = 1.e-10
			fHxc[i1,i2,t] = fHxc[i1,i2,t] / (chiH.chi_oft[i1,i2,t] * chiKS.chi_oft[i1,i2,t])
			
#### write output file ###########################
outf = outdir + "chi11.dat"
f = open(outf, 'w')
for i in range(chiH.Nt):
	f.write( "%.10f        " % chiH.time[i] + "%.10f        " % chiH.chi_oft[0,0,i].real + "%.10f        " % chiH.chi_oft[0,0,i].imag + "%.10f        " % chiKS.chi_oft[0,0,i].real + "%.10f" % chiKS.chi_oft[0,0,i].imag + "\n" )
f.close()

outf = outdir + "chi12.dat"
f = open(outf, 'w')
for i in range(chiH.Nt):
	f.write( "%.10f        " % chiH.time[i] + "%.10f        " % chiH.chi_oft[0,1,i].real + "%.10f        " % chiH.chi_oft[0,1,i].imag + "%.10f        " % chiKS.chi_oft[0,1,i].real + "%.10f" % chiKS.chi_oft[0,1,i].imag + "\n" )
f.close()

outf = outdir + "chi21.dat"
f = open(outf, 'w')
for i in range(chiH.Nt):
	f.write( "%.10f        " % chiH.time[i] + "%.10f        " % chiH.chi_oft[1,0,i].real + "%.10f        " % chiH.chi_oft[1,0,i].imag + "%.10f        " % chiKS.chi_oft[1,0,i].real + "%.10f" % chiKS.chi_oft[1,0,i].imag + "\n" )
f.close()

outf = outdir + "chi22.dat"
f = open(outf, 'w')
for i in range(chiH.Nt):
	f.write( "%.10f        " % chiH.time[i] + "%.10f        " % chiH.chi_oft[1,1,i].real + "%.10f        " % chiH.chi_oft[1,1,i].imag + "%.10f        " % chiKS.chi_oft[1,1,i].real + "%.10f" % chiKS.chi_oft[1,1,i].imag + "\n" )
f.close()

##### write output for XC kernel in time ##########
outf = outdir + "fhxc11.dat"
f = open(outf, 'w')
for i in range(chiH.Nt):
	f.write( "%.10f        " % chiH.time[i] + "%.10f        " % fHxc[0,0,i].real + "%.10f        " % fHxc[0,0,i].imag + "\n" )
f.close()

outf = outdir + "fhxc12.dat"
f = open(outf, 'w')
for i in range(chiH.Nt):
	f.write( "%.10f        " % chiH.time[i] + "%.10f        " % fHxc[0,1,i].real + "%.10f        " % fHxc[0,1,i].imag + "\n" )
f.close()

outf = outdir + "fhxc21.dat"
f = open(outf, 'w')
for i in range(chiH.Nt):
	f.write( "%.10f        " % chiH.time[i] + "%.10f        " % fHxc[1,0,i].real + "%.10f        " % fHxc[1,0,i].imag + "\n" )
f.close()

outf = outdir + "fhxc22.dat"
f = open(outf, 'w')
for i in range(chiH.Nt):
	f.write( "%.10f        " % chiH.time[i] + "%.10f        " % fHxc[1,1,i].real + "%.10f        " % fHxc[1,1,i].imag + "\n" )
f.close()

sys.exit()
