#
#     This script create a dataset
#     for td Hubbard evolution
#     
#     run over different external potential for dimer
#     	1) first step : create input2 file
#       2) second step : run TD Hubbard model
#       3) collect result 
#     output file :
#     for each simulation output file with 4 columns
#     1) v_ext_1(t)   2) v_ext_2(t)   3) E(t)   4) n_1(t)   5) n_2(t)    6) F[n](t)
#
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import Set_Hamilt
import scipy.linalg as la
import expect_val
import time_integration
#
#     vext(i,t) = v(i) + C * exp(-(t-t0)^2/2/tau^2) * sin(a*(t-t0)+phi(i))
#
#     parameters to define : a(i), v(i), t0, tau
#
#     vext_i(t) = vext_i[a,v,t0,tau](t)
#
print("...... READ INPUT FILE .......")
#
#     read input0 file
#
f = open("input_scr", 'r')
lines = f.readlines()
for line in lines:
	l = line.strip()
	l = line.split()
	if len(l) > 0:
		#### outdir ###############
		if l[0] == "outdir":
			outdir = l[2]
		#### calculation ##########
		if l[0] == "calc":
			calc = l[2]
			if calc != "Hubbard":
				raise ValueError("calc variable is not set to Hubbard")
		#### H parameters #########
		if l[0] == "t":
			t = float(l[2])
		if l[0] == "U":
			U = float(l[2])
		#### potential data #######
		if l[0] == "vst":
			vst = np.zeros(2)
			vst[0] = float(l[2])
			vst[1] = float(l[4])
		if l[0] == "vfin":
			vfin = np.zeros(2)
			vfin[0] = float(l[2])
			vfin[1] = float(l[4])
		#### input parameters #####
		if l[0] == "nv":
			nv = int(l[2])
		if l[0] == "na":
			na = int(l[2])
		if l[0] == "a_in":
			a_in = float(l[2])
		if l[0] == "a_fin":
			a_fin = float(l[2])
		if l[0] == "nt0":
			nt0 = int(l[2])
		if l[0] == "t0i":
			t0i = float(l[2])
		if l[0] == "dt0":
			dt0 = float(l[2])
		if l[0] == "ntau":
			ntau = int(l[2])
		if l[0] == "dtau":
			dtau = float(l[2])
		if l[0] == "tau0":
			tau0 = float(l[2])
		if l[0] == "nC":
			nC = int(l[2])
		if l[0] == "nphi":
			nphi = int(l[2])
		### td evolution parameters ##
		if l[0] == "T":
			T = float(l[2])
		if l[0] == "dt":
			dt = float(l[2])
#
nsim = na*nt0*ntau*nv*nC
print("total number simulations= ", nsim)
#
#     run cycle - C values
#
dC = 1.
C_values = np.zeros(nC)
for iC in range(nC):
	C_values[iC] = iC * dC
#
#     run cycle v - external potentials
#
dv = abs (vfin[0] - vst[0]) / nv
print("dv= ", dv)
v_list = np.zeros((nv, 2))
dv_values = np.zeros(nv)
#
for iv in range(nv):
	v_list[iv,0] = vst[0] + iv*dv
	v_list[iv,1] = vst[1] - iv*dv
	dv_values[iv] = v_list[iv,0] - v_list[iv,1]
#
#     run t0 cycle
#
t0_values = np.zeros(nt0)
for it in range(nt0):
	t0_values[it] = t0i + it * dt0
#
#     run tau cycle
#
tau_values = np.zeros(ntau)
for it in range(ntau):
	tau_values[it] = tau0 + it * dtau
#
#     run a cycle
#
a_values = np.zeros(na)
da = (a_fin - a_in)/(na-1)
for ia in range(na):
	a_values[ia] = a_in + ia * da
#
#     run phase cycle
#
phi_values = np.linspace(0, np.pi/2, nphi)
#
#     set list of external potentials
#
#     set time array
time = np.arange(0., T+dt, dt)
Nt = len(time)
print("n. time steps= ", Nt)
time2= np.arange(0., T+dt/2, dt/2)
Nt2 = len(time2)
#
#  define external potential
#
vext_t = np.zeros((2,Nt))
vext_t2 = np.zeros((2,Nt2))
#
#  data frame columns
#
columns = ['Delta vext_t', 'Delta n_t', 'E_t', 'KH_t']
#
#     iterate over different parameters
#
j = 1
for iv in range(nv):
	v = np.zeros(2)
	v[:] = v_list[iv,:]
	for iC in range(nC):
		C = C_values[iC]
		if C == 0:
			vext_t[:,:] = 0.
			vext_t[0,:] = v[0]
			vext_t[1,:] = v[1]
			#
			vext_t2[:,:] = 0.
			vext_t2[0,:] = v[0]
			vext_t2[1,:] = v[1]
		else:
			for it0 in range(nt0):
				t0 = t0_values[it0]
				for itau in range(ntau):
					tau = tau_values[itau]
					for ia in range(na):
						a = a_values[ia]
						for iphi in range(nphi):
							phi = phi_values[iphi]
							#
							vext_t[:,:] = 0.
							vext_t[0,:] = v[0] + C * np.exp(-(time[:]-t0)**2/2./tau**2) * np.sin(a*(time[:]-t0))
							vext_t[1,:] = v[1] + C * np.exp(-(time[:]-t0)**2/2./tau**2) * np.sin(a*(time[:]-t0)+phi)
							#plt.plot(time, vext_t[0,:])
							#plt.plot(time, vext_t[1,:])
							#plt.show()
							vext_t2[:,:] = 0.
							vext_t2[0,:] = v[0] + C * np.exp(-(time2[:]-t0)**2/2./tau**2) * np.sin(a*(time2[:]-t0))
							vext_t2[1,:] = v[1] + C * np.exp(-(time2[:]-t0)**2/2./tau**2) * np.sin(a*(time2[:]-t0)+phi)
							if abs(vext_t[0,0]-vext_t[0,1]) > 1.E-7 or abs(vext_t[1,0]-vext_t[1,1]) > 1.E-7:
								pass
							else:
								#
								#  run Hubbard model evolution
								#
								#  set up the Hamiltonian
								#
								data = {}
								vext = np.zeros(2)
								vext[:] = vext_t[:,0]
								H = Set_Hamilt.Hubbard_Hamilt_twosites( t, U, vext )
								#
								#  compute the Hamiltonian ground state
								#
								[eig, eigv] = la.eigh(H)
								eig = eig.real
								i = int(np.where(eig == min(eig))[0])
								wf0 = eigv[:,i]
								#  compute GS occupations
								n = expect_val.gs_occup( calc, eig, eigv )
								# energy expect. value
								E = expect_val.H_expect_gs( calc, eig, U, t, vext, n )
								# start td evolution
								# use RK4 algorithm
								tdocc, Eoft = time_integration.RK4( dt, T, wf0, calc, vext_t2, t, U )
								# compute KH functional
								KH_Func = np.zeros(len(Eoft))
								for it in range(len(Eoft)):
									vext[:] = vext_t[:,it]
									vn = np.dot(vext, tdocc[:,it])
									KH_Func[it] = Eoft[it] - vn
								# collect data
								delta_n = np.zeros(Eoft.shape)
								delta_n[:] = tdocc[0,:] - tdocc[1,:]
								delta_vext = np.zeros(Eoft.shape)
								for it in range(len(Eoft)):
									delta_vext[it] = vext_t[0,it] - vext_t[1,it]
								data = dict(zip(columns, [delta_vext, delta_n, Eoft, KH_Func]))
								# build data frame
								data_frame = pd.DataFrame(data=data)
								data_frame.to_hdf('data.h5', key='df'+str(j), mode="a")
								j = j+1
#
#  add time to data
#
t = pd.Series(time)
t.to_hdf('data.h5', key='time', mode="a")
sys.exit()
