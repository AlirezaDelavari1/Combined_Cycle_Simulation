#Import Libraries
import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import os
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from pymoo.optimize import minimize


#Initial conditions

fluid_1 = "Air"
fluid_2 = "Water"
m_dot_air = 50 #kg/s
m_dot_water = 3 #kg/s
T0 = 298.15 #Kelvin
P0 = 1.01325e5 #Pa
P11 = 4e6 #Pa
P15 = 10e3 #Pa
T_max = T3 =  1100 #Kelvin | T @ Gas Turbine's inlet
h0 = CP.PropsSI("H", "P", P0, "T", T0, fluid_1)
s0 = CP.PropsSI("S", "P", P0, "T", T0, fluid_1)

#Fuel specification

#Fuel: Methane (CH4)
LHV = 50e6

#Number of flows in system

m = 17 

#Equipment specification

#Compressor

rp_comp = 5
eta_isen_comp = 0.8

#Combustion chamber

dp_cc = 0.01
eta_th_cc = 0.98

#Gas turbine

rp_gt = 4.5
eta_isen_gt = 0.89

#Heat exchanger 1

epsilon_HX1 = 0.95
eta_th_HX1 = 1
dp_HX1 = 0.02

#Heat exchanger 2

eta_th_HX2 = 1
dp_HX2 = 0.03

#Steam turbine

eta_isen_st = 0.95

#Condenser

eta_th_cond = 1
dp_cond = 0.02

#Pump

eta_isen_pump = 0.98

class combined_cycle():

    def __init__(self, fluid_1, fluid_2, m_dot_air, m_dot_water, T0, P0, P11, P15, T3, m, rp_comp, eta_isen_comp, dp_cc, eta_th_cc, rp_gt, eta_isen_gt, epsilon_HX1,
                 eta_th_HX1, dp_HX1, eta_th_HX2, dp_HX2, eta_isen_st, eta_th_cond, dp_cond, eta_isen_pump):
        
        self.fluid_1 = fluid_1
        self.fluid_2 = fluid_2
        self.m_dot_air = m_dot_air
        self.m_dot_water = m_dot_water
        self.T0 = T0
        self.P0 = P0
        self.P11 = P11
        self.P15 = P15
        self.T3 = T3
        self.m = m
        self.rp_comp = rp_comp
        self.eta_isen_comp = eta_isen_comp
        self.dp_cc = dp_cc
        self.eta_th_cc = eta_th_cc
        self.rp_gt = rp_gt
        self.eta_isen_gt = eta_isen_gt
        self.epsilon_HX1 = epsilon_HX1
        self.eta_th_HX1 = eta_th_HX1
        self.dp_HX1 = dp_HX1
        self.eta_th_HX2 = eta_th_HX2
        self.dp_HX2 = dp_HX2
        self.eta_isen_st = eta_isen_st
        self.eta_th_cond = eta_th_cond
        self.dp_cond = dp_cond
        self.eta_isen_pump = eta_isen_pump

    def pressure(self):
        #Pressure matrix

        P=np.zeros(self.m)

        P[0] = self.P0
        P[1] = P[0]*self.rp_comp
        P[2] = P[1]*(1-self.dp_HX1)
        P[3] = P[2]*(1-self.dp_cc)
        P[7] = P[3]/self.rp_gt
        P[8] = P[7]*(1-self.dp_HX2)
        P[9] = P[8]*(1-self.dp_HX1)
        P[11] = self.P11
        P[10] = P[11]*(1+self.dp_HX1)
        P[15] = self.P15
        P[13] = P[15]/(1-self.dp_cond)

        return P
    

    def energy_balance(self):

        P = self.pressure()


        h0 = CP.PropsSI("H", "P", self.P0, "T", self.T0, self.fluid_1)
        s0 = CP.PropsSI("S", "P", self.P0, "T", self.T0, self.fluid_1)

        s1 = s0
        h1s = CP.PropsSI("H", "P", P[1], "S", s1, self.fluid_1)

        h3 = CP.PropsSI("H", "P", P[3], "T", self.T3, self.fluid_1)
        s3 = CP.PropsSI("S", "P", P[3], "T", self.T3, self.fluid_1)

        s7 = s3
        h7s = CP.PropsSI("H", "P", P[7], "S", s7, self.fluid_1)

        h11 = CP.PropsSI("H", "P", P[11], "Q", 1, self.fluid_2)
        s11 = CP.PropsSI("S", "P", P[11], "Q", 1, self.fluid_2)
        s13 = s11
        h13s = CP.PropsSI("H", "P", P[13], "S", s13, self.fluid_2)

        h15 = CP.PropsSI("H", "P", P[15], "Q", 0, self.fluid_2)
        s15 = CP.PropsSI("S", "P", P[15], "Q", 0, self.fluid_2)
        s10 = s15
        h10s = CP.PropsSI("H", "P", P[10], "S", s10, self.fluid_2)

        #Energy balance

        A=np.zeros((self.m,self.m))
        X=np.zeros(self.m)      #X=[h0,h1,h2,h3,Q_cc,w_gt,w_comp,h7,h8,h9,h10,h11,w_st,h13,Q_cond,h15,w_pump]
        B=np.zeros(self.m)

        #Compressor
        A[0,0] = self.m_dot_air
        A[0,1] = -self.m_dot_air
        A[0,6] = 1
        B[0] = 0

        #HX1
        A[1,1] = self.m_dot_air
        A[1,2] = -self.m_dot_air
        A[1,8] = self.m_dot_air
        A[1,9] = -self.m_dot_air
        B[1] = 0

        #CC
        A[2,2] = self.m_dot_air
        A[2,3] = -self.m_dot_air
        A[2,4] = 1
        B[2] = 0

        #GT
        A[3,3] = self.m_dot_air
        A[3,5] = -1
        A[3,6] = -1
        A[3,7] = -self.m_dot_air
        B[3] = 0

        #HX2
        A[4,7] = self.m_dot_air
        A[4,8] = -self.m_dot_air
        A[4,10] = self.m_dot_water
        A[4,11] = -self.m_dot_water
        B[4] = 0

        #ST
        A[5,11] = self.m_dot_water
        A[5,12] = -1
        A[5,13] = -self.m_dot_water
        B[5] = 0

        #Condenser
        A[6,13] = self.m_dot_water
        A[6,14] = -1
        A[6,15] = -self.m_dot_water
        B[6] = 0

        #Pump
        A[7,10] = -self.m_dot_water
        A[7,15] = self.m_dot_water
        A[7,16] = 1
        B[7] = 0

        #Isentropic efficiency of Compressor
        A[8,0] = 1-self.eta_isen_comp
        A[8,1] = self.eta_isen_comp
        B[8] = h1s

        #Isentropic efficiency of Gas turbine
        A[9,3] = 1-(1/self.eta_isen_gt)
        A[9,7] = 1/self.eta_isen_gt
        B[9] = h7s

        #Isentropic efficiency of Steam turbine
        A[10,11] = 1-(1/self.eta_isen_st)
        A[10,13] = 1/self.eta_isen_st
        B[10] = h13s

        #Isentropic efficiency of Pump
        A[11,15] = 1-self.eta_isen_pump
        A[11,10] = self.eta_isen_pump
        B[11] = h10s

        #Effectiveness of HX1
        A[12,1] = 1-self.epsilon_HX1
        A[12,2] = -1
        A[12,8] = self.epsilon_HX1
        B[12] = 0

        #known variables
        #based on T0 and P0
        A[13,0] = 1
        B[13] = h0

        #based on T3 and P3
        A[14,3] = 1
        B[14] = h3

        #based on P15 and X
        A[15,15] = 1
        B[15] = h15

        #based on P11 and X
        A[16,11] = 1
        B[16] = h11

        
        X = np.linalg.solve(A, B)

        return X
    
    def turbines_works(self):
        X = self.energy_balance()

        GT_work = X[5]
        ST_work = X[12]

        return [GT_work, ST_work]
    
    def comp_pump_works(self):
    
        X = self.energy_balance()

        comp_work = X[6]
        pump_work = X[16]

        return [comp_work, pump_work]

    

    def enthalpy(self):
        #Enthalpy
        X = self.energy_balance()
        enthalpy = np.zeros(self.m) #J/kg
        for i in range (self.m):
            if i in (4,5,6,12,14,16): 
                continue
            enthalpy[i] = X[i]

        return enthalpy
    
    def temperature(self):
        #Temperature
        P = self.pressure()
        X = self.energy_balance()

        T = np.zeros(self.m) #Kelvin
        for i in range (0,10):
            if i not in (4,5,6):
                T[i]= CP.PropsSI("T", "P", P[i], "H", X[i], self.fluid_1)
        for i in range (10,16):
            if i not in (12,14,16):
                T[i]= CP.PropsSI("T", "P", P[i], "H", X[i], self.fluid_2)

        return T
    
    def entropy(self):
        P = self.pressure()
        X = self.energy_balance()
        s = np.zeros(self.m)
        for i in range(self.m):
            if i in (4,5,6,12,14,16): 
                continue 
            elif i in (0,1,2,3,7,8,9):
                s[i] = CP.PropsSI("S", "P", P[i], "H", X[i], self.fluid_1)
            elif i in (10,11,13,15):
                s[i] = CP.PropsSI("S", "P", P[i], "H", X[i], self.fluid_2)
            else:
                print("Something is wrong!!!")
        return s
        


    def m_dot_fuel(self):
        P = self.pressure()
        X = self.energy_balance()
        Q_CC = X[4]
        m_dot_fuel = Q_CC / (eta_th_cc * LHV) 

        return m_dot_fuel
    
    def exergy(self):
        P = self.pressure()
        X = self.energy_balance()
        s = self.entropy()

        EX = np.zeros(self.m)
        for i in range(0,10):
            if i in (4,5,6):
                EX[i] = 0
            else:
                EX[i] = self.m_dot_air * ((X[i]-X[0]) - self.T0*(s[i]-s[0]))
            
        for i in range(10,16):
            if i in (12,14,16): 
                EX[i] = 0
            else:
                EX[i] = self.m_dot_water * ((X[i]-X[0]) - self.T0*(s[i]-s[0]))

            
        EX[5] = X[5]
        EX[6] = X[6]
        EX[12] = X[12]
        EX[16] = X[16]

        ex_ch_fuel = 51.94e6 #MJ/Kg #Methane

        EX[4] = self.m_dot_fuel() * ex_ch_fuel

        T13 = CP.PropsSI("T", "P", P[13], "H", X[13], self.fluid_2)
        T15 = CP.PropsSI("T", "P", P[15], "H", X[15], self.fluid_2)
        T_avg_cond = (T13 + T15)/2

        EX[14] = (1 - self.T0/T_avg_cond) * X[14]

        return EX
    
    def eta_th_plant(self):
        P = self.pressure()
        X = self.energy_balance()
        s = self.entropy()
        W_net = X[5] + X[12] - X[16]

        eta_th_plant = (W_net / (X[4])) * 100
        
        return eta_th_plant
    
    def exergy_des(self):
        P = self.pressure()
        X = self.energy_balance()
        s = self.entropy()
        EX = self.exergy()

        EX_Des = np.zeros(8)
        
        #Comp
        EX_Des[0] = EX[0] - EX[1] + EX[6]

        #HX1
        EX_Des[1] = EX[1] + EX[8] - EX[2] - EX[9]

        #CC
        EX_Des[2] = EX[2] - EX[3] + EX[4]

        #GT
        EX_Des[3] = EX[3] - EX[7] - EX[5]

        #HX2
        EX_Des[4] = EX[7] - EX[8] + EX[10] - EX[11]

        #ST
        EX_Des[5] = EX[11] - EX[13] - EX[12]

        #Cond
        EX_Des[6] = EX[13] - EX[15] - EX[14]

        #Pump
        EX_Des[7] = EX[15] - EX[10] + EX[16]
        
        return EX_Des
    

    def exergy_efficiency(self):
        P = self.pressure()
        X = self.energy_balance()
        s = self.entropy()
        EX = self.exergy()

        EX_eff = np.zeros(8)
    
        #Comp
        EX_eff[0] = ((EX[1]-EX[0]) / EX[6]) * 100

        #HX1
        EX_eff[1] = (EX[2]-EX[1]) / (EX[8]-EX[9]) * 100

        #CC
        EX_eff[2] = ((EX[3] - EX[2]) / EX[4]) * 100

        #GT
        EX_eff[3] = (EX[5] / (EX[3] - EX[7])) * 100

        #HX2
        EX_eff[4] = ((EX[11] - EX[10]) / (EX[7] - EX[8])) * 100

        #ST
        EX_eff[5] = (EX[12] / (EX[11] - EX[13])) * 100

        #Cond
        EX_eff[6] = (EX[15] / EX[13]) * 100

        #Pump
        EX_eff[7] = ((EX[10] - EX[15]) / EX[16]) * 100
    
        return EX_eff
    
    def plant_EX_Des(self):
        P = self.pressure()
        X = self.energy_balance()
        s = self.entropy()
        EX = self.exergy()
        EX_Des = self.exergy_des()

        Sum_EX = sum(EX_Des)

        return Sum_EX
    
    def plant_EX_eff(self):
        P = self.pressure()
        X = self.energy_balance()
        s = self.entropy()
        EX = self.exergy()
        EX_Des = self.exergy_des()

        return ((EX[5] + EX[12] - EX[16]) / EX[4]) * 100
    
combined_cycle_power_plant_model = combined_cycle(fluid_1, fluid_2, m_dot_air, m_dot_water, T0, P0, P11, P15, T3, m, rp_comp, eta_isen_comp, dp_cc, eta_th_cc, rp_gt, eta_isen_gt, epsilon_HX1,
                 eta_th_HX1, dp_HX1, eta_th_HX2, dp_HX2, eta_isen_st, eta_th_cond, dp_cond, eta_isen_pump)

print("pressure Matrix: \n", combined_cycle_power_plant_model.pressure())
print("energy_balance Matrix: \n",combined_cycle_power_plant_model.energy_balance())
print("enthalpy Matrix: \n",combined_cycle_power_plant_model.enthalpy())
print("temperature Matrix: \n",combined_cycle_power_plant_model.temperature())
print("entropy Matrix: \n",combined_cycle_power_plant_model.entropy())
print("m_dot_fuel: \n",combined_cycle_power_plant_model.m_dot_fuel())
print("exergy Matrix: \n",combined_cycle_power_plant_model.exergy())
print("eta_th_plant: \n",combined_cycle_power_plant_model.eta_th_plant())
print("exergy_des Matrix: \n",combined_cycle_power_plant_model.exergy_des())
print("exergy_efficiency Matrix: \n",combined_cycle_power_plant_model.exergy_efficiency()) 
print("plant_EX_Des: \n",combined_cycle_power_plant_model.plant_EX_Des())
print("plant_EX_eff: \n",combined_cycle_power_plant_model.plant_EX_eff())
print("turbines_works Matrix(GT and ST): \n",combined_cycle_power_plant_model.turbines_works())
print("comp_pump_works Matrix(comp and pump): \n",combined_cycle_power_plant_model.comp_pump_works())

#Plots (Variables vs Plant Exergy Efficiency)

#(plant_EX_eff vs eta_isen_GT)
eta_isen_GT_range = np.arange(.5,1,.01)


plant_EX_eff_List = []

for eta in eta_isen_GT_range:

    combined_cycle_power_plant_model_plot = combined_cycle(fluid_1, fluid_2, m_dot_air, m_dot_water, T0, P0, P11, P15, T3, m, rp_comp, eta_isen_comp, dp_cc, eta_th_cc, rp_gt, eta, epsilon_HX1,
                 eta_th_HX1, dp_HX1, eta_th_HX2, dp_HX2, eta_isen_st, eta_th_cond, dp_cond, eta_isen_pump)
    
    efficiency = combined_cycle_power_plant_model_plot.plant_EX_eff()
    plant_EX_eff_List.append(efficiency)

    print(f"eta_isen_GT: {eta}, Plant Exergy Efficiency: {efficiency}")


plt.plot(eta_isen_GT_range, plant_EX_eff_List, color="orange", marker='o', linestyle='-', linewidth=2, markersize=5)

plt.title("Exergy Efficiency vs Gas Turbine Isentropic Efficiency", fontsize=16)
plt.xlabel("Isentropic Efficiency of Gas Turbine (%)", fontsize=14)
plt.ylabel("Plant Exergy Efficiency (%)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
# current_directory = os.getcwd()
# print(f"Current working directory: {current_directory}")
# file_path = os.path.join(current_directory, 'etaGT_plantEXeff.png')
# try:
#     plt.savefig(file_path, dpi=300, bbox_inches='tight')
#     print(f"Figure saved successfully at: {file_path}")
# except Exception as e:
#     print(f"Error saving figure: {e}")
plt.show()

# #(plant_EX_eff vs eta_isen_ST)
eta_isen_ST_range = np.arange(.5,1,.01)


plant_EX_eff_List2 = []

for eta in eta_isen_ST_range:

    combined_cycle_power_plant_model_plot = combined_cycle(fluid_1, fluid_2, m_dot_air, m_dot_water, T0, P0, P11, P15, T3, m, rp_comp, eta_isen_comp, dp_cc, eta_th_cc, rp_gt, eta_isen_gt, epsilon_HX1,
                 eta_th_HX1, dp_HX1, eta_th_HX2, dp_HX2, eta, eta_th_cond, dp_cond, eta_isen_pump)
    
    efficiency = combined_cycle_power_plant_model_plot.plant_EX_eff()
    plant_EX_eff_List2.append(efficiency)

    print(f"eta_isen_ST: {eta}, Plant Exergy Efficiency: {efficiency}")


plt.plot(eta_isen_ST_range, plant_EX_eff_List2, color="orange", marker='o', linestyle='-', linewidth=2, markersize=5)

plt.title("Exergy Efficiency vs Steam Turbine Isentropic Efficiency", fontsize=16)
plt.xlabel("Isentropic Efficiency of Steam Turbine (%)", fontsize=14)
plt.ylabel("Plant Exergy Efficiency (%)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
# current_directory = os.getcwd()
# print(f"Current working directory: {current_directory}")
# file_path = os.path.join(current_directory, 'etaST_plantEXeff.png')
# try:
#     plt.savefig(file_path, dpi=300, bbox_inches='tight')
#     print(f"Figure saved successfully at: {file_path}")
# except Exception as e:
#     print(f"Error saving figure: {e}")
plt.show()

# #(plant_EX_eff vs m_dot_water)
m_dot_water_range = np.arange(.1,6,.1)


plant_EX_eff_List3 = []

for mdot in m_dot_water_range:

    combined_cycle_power_plant_model_plot = combined_cycle(fluid_1, fluid_2, m_dot_air, mdot, T0, P0, P11, P15, T3, m, rp_comp, eta_isen_comp, dp_cc, eta_th_cc, rp_gt, eta_isen_gt, epsilon_HX1,
                 eta_th_HX1, dp_HX1, eta_th_HX2, dp_HX2, eta_isen_st, eta_th_cond, dp_cond, eta_isen_pump)
    
    efficiency = combined_cycle_power_plant_model_plot.plant_EX_eff()
    plant_EX_eff_List3.append(efficiency)

    print(f"m_dot_water: {mdot}, Plant Exergy Efficiency: {efficiency}")


plt.plot(m_dot_water_range, plant_EX_eff_List3, color="orange", marker='o', linestyle='-', linewidth=2, markersize=5)

plt.title("Exergy Efficiency vs M dot of Water", fontsize=16)
plt.xlabel("M dot of Water (Kg/s)", fontsize=14)
plt.ylabel("Plant Exergy Efficiency (%)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=12)
# current_directory = os.getcwd()
# print(f"Current working directory: {current_directory}")
# file_path = os.path.join(current_directory, 'mdotWater_plantEXeff.png')
# try:
#     plt.savefig(file_path, dpi=300, bbox_inches='tight')
#     print(f"Figure saved successfully at: {file_path}")
# except Exception as e:
#     print(f"Error saving figure: {e}")
plt.show()

# #(plant_EX_eff vs rp_comp)
rp_comp_range = np.arange(2,9,.1)


plant_EX_eff_List4 = []

for rp in rp_comp_range:

    combined_cycle_power_plant_model_plot = combined_cycle(fluid_1, fluid_2, m_dot_air, m_dot_water, T0, P0, P11, P15, T3, m, rp, eta_isen_comp, dp_cc, eta_th_cc, rp_gt, eta_isen_gt, epsilon_HX1,
                 eta_th_HX1, dp_HX1, eta_th_HX2, dp_HX2, eta_isen_st, eta_th_cond, dp_cond, eta_isen_pump)
    
    efficiency = combined_cycle_power_plant_model_plot.plant_EX_eff()
    plant_EX_eff_List4.append(efficiency)

    print(f"rp_comp: {rp}, Plant Exergy Efficiency: {efficiency}")


plt.plot(rp_comp_range, plant_EX_eff_List4, color="orange", marker='o', linestyle='-', linewidth=2, markersize=5)

plt.title("Exergy Efficiency vs Rp of Compressor", fontsize=16)
plt.xlabel("Rp of Compressor", fontsize=14)
plt.ylabel("Plant Exergy Efficiency (%)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=12)
# current_directory = os.getcwd()
# print(f"Current working directory: {current_directory}")
# file_path = os.path.join(current_directory, 'rpComp_plantEXeff.png')
# try:
#     plt.savefig(file_path, dpi=300, bbox_inches='tight')
#     print(f"Figure saved successfully at: {file_path}")
# except Exception as e:
#     print(f"Error saving figure: {e}")
plt.show()

# #(plant_EX_eff vs P11)
P11_range = np.arange(2e6,6e6,1e5)


plant_EX_eff_List5 = []

for P in P11_range:

    combined_cycle_power_plant_model_plot = combined_cycle(fluid_1, fluid_2, m_dot_air, m_dot_water, T0, P0, P, P15, T3, m, rp_comp, eta_isen_comp, dp_cc, eta_th_cc, rp_gt, eta_isen_gt, epsilon_HX1,
                 eta_th_HX1, dp_HX1, eta_th_HX2, dp_HX2, eta_isen_st, eta_th_cond, dp_cond, eta_isen_pump)
    
    efficiency = combined_cycle_power_plant_model_plot.plant_EX_eff()
    plant_EX_eff_List5.append(efficiency)

    print(f"P11: {P}, Plant Exergy Efficiency: {efficiency}")


plt.plot(P11_range, plant_EX_eff_List5, color="orange", marker='o', linestyle='-', linewidth=2, markersize=5)

plt.title("Exergy Efficiency vs Pressure values of stream 11", fontsize=16)
plt.xlabel("Pressure values of stream 11 (Pa)", fontsize=14)
plt.ylabel("Plant Exergy Efficiency (%)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=12)
# current_directory = os.getcwd()
# print(f"Current working directory: {current_directory}")
# file_path = os.path.join(current_directory, 'P11_plantEXeff.png')
# try:
#     plt.savefig(file_path, dpi=300, bbox_inches='tight')
#     print(f"Figure saved successfully at: {file_path}")
# except Exception as e:
#     print(f"Error saving figure: {e}")
plt.show()

# #(plant_EX_eff vs T3)
T3_range = np.arange(500,1700,20)


plant_EX_eff_List6 = []

for T in T3_range:

    combined_cycle_power_plant_model_plot = combined_cycle(fluid_1, fluid_2, m_dot_air, m_dot_water, T0, P0, P11, P15, T, m, rp_comp, eta_isen_comp, dp_cc, eta_th_cc, rp_gt, eta_isen_gt, epsilon_HX1,
                 eta_th_HX1, dp_HX1, eta_th_HX2, dp_HX2, eta_isen_st, eta_th_cond, dp_cond, eta_isen_pump)
    
    efficiency = combined_cycle_power_plant_model_plot.plant_EX_eff()
    plant_EX_eff_List6.append(efficiency)

    print(f"T3: {T}, Plant Exergy Efficiency: {efficiency}")


plt.plot(T3_range, plant_EX_eff_List6, color="orange", marker='o', linestyle='-', linewidth=2, markersize=5)

plt.title("Exergy Efficiency vs Temperature values of stream 3", fontsize=16)
plt.xlabel("Temperature values of stream 3 (K)", fontsize=14)
plt.ylabel("Plant Exergy Efficiency (%)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=12)
# current_directory = os.getcwd()
# print(f"Current working directory: {current_directory}")
# file_path = os.path.join(current_directory, 'T3_plantEXeff.png')
# try:
#     plt.savefig(file_path, dpi=300, bbox_inches='tight')
#     print(f"Figure saved successfully at: {file_path}")
# except Exception as e:
#     print(f"Error saving figure: {e}")
plt.show()

#(plant_EX_eff vs dp_cc)
dp_cc_range = np.arange(0.01,0.99,0.01)


plant_EX_eff_List7 = []

for dp_c in dp_cc_range:

    combined_cycle_power_plant_model_plot = combined_cycle(fluid_1, fluid_2, m_dot_air, m_dot_water, T0, P0, P11, P15, T3, m, rp_comp, eta_isen_comp, dp_c, eta_th_cc, rp_gt, eta_isen_gt, epsilon_HX1,
                 eta_th_HX1, dp_HX1, eta_th_HX2, dp_HX2, eta_isen_st, eta_th_cond, dp_cond, eta_isen_pump)
    
    efficiency = combined_cycle_power_plant_model_plot.plant_EX_eff()
    plant_EX_eff_List7.append(efficiency)

    print(f"dp_cc: {dp_c}, Plant Exergy Efficiency: {efficiency}")


plt.plot(dp_cc_range, plant_EX_eff_List7, color="orange", marker='o', linestyle='-', linewidth=2, markersize=5)

plt.title("Exergy Efficiency vs Differential Pressure of Combustion Chamber", fontsize=16)
plt.xlabel("Differential Pressure of Combustion Chamber", fontsize=14)
plt.ylabel("Plant Exergy Efficiency (%)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=12)
# current_directory = os.getcwd()
# print(f"Current working directory: {current_directory}")
# file_path = os.path.join(current_directory, 'dp_cc_plantEXeff.png')
# try:
#     plt.savefig(file_path, dpi=300, bbox_inches='tight')
#     print(f"Figure saved successfully at: {file_path}")
# except Exception as e:
#     print(f"Error saving figure: {e}")
plt.show()

#Optimization

GA_Alg = NSGA2(
    pop_size=150,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

class Exergy_Opt_problem(ElementwiseProblem):

    #decision variables = [m_dot_water, eta_isen_gt, eta_isen_st, rp_comp, P11]
    def __init__(self):
        super().__init__(n_var=5, n_obj=2, xl=np.array([0.1, 0.7, 0.7, 2, 2e6]), xu=np.array([6, 0.99, 0.99, 9, 6e6]))

    def _evaluate(self, x, out):

        combined_cycle_power_plant_evaluate = combined_cycle(fluid_1, fluid_2, m_dot_air, x[0], T0, P0, x[4], P15, T3, m, x[3], eta_isen_comp, dp_cc, eta_th_cc, rp_gt, x[1], epsilon_HX1,
                 eta_th_HX1, dp_HX1, eta_th_HX2, dp_HX2, x[2], eta_th_cond, dp_cond, eta_isen_pump)

        f1 = combined_cycle_power_plant_evaluate.plant_EX_eff()
        f2 = combined_cycle_power_plant_evaluate.m_dot_fuel()

        out["F"]= np.array([-f1,f2])

        return super()._evaluate(x, out)

prob=Exergy_Opt_problem()

term = get_termination("n_gen",35)

res = minimize(prob,
               GA_Alg,
               term,
               seed=1,
               save_history=True,
               verbose=True)

X = res.X
F = res.F

plt.scatter(F[:,0],F[:,1], color='blue', marker='o', s=50, label='Pareto Front')
plt.title('Pareto Front of Multi-Objective Optimization', fontsize=16)
plt.xlabel('Plant Exergy Efficiency (%)', fontsize=14)
plt.ylabel('M dot of Fuel (Kg/s)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
# current_directory = os.getcwd()
# print(f"Current working directory: {current_directory}")
# file_path = os.path.join(current_directory, 'pareto_front.png')
# try:
#     plt.savefig(file_path, dpi=300, bbox_inches='tight')
#     print(f"Figure saved successfully at: {file_path}")
# except Exception as e:
#     print(f"Error saving figure: {e}")
plt.show()

F=list(F)
for i in range(len(F)):
    F[i]=list(F[i])

X=list(X)
for i in range(len(X)):
    X[i]=list(X[i])


F_array = np.array(F)

#Get the maximum value of the first element in F
max_first_element_index = np.argmin(F_array[:, 0])
max_first_element_value = F_array[max_first_element_index, 0]
corresponding_X_first = X[max_first_element_index]

print(f"Max first element: {max_first_element_value}, Corresponding X: {corresponding_X_first}")

#Get the lowest value of the second element in F
min_second_element_index = np.argmin(F_array[:, 1])
min_second_element_value = F_array[min_second_element_index, 1]
corresponding_X_second = X[min_second_element_index]

print(f"Min second element: {min_second_element_value}, Corresponding X: {corresponding_X_second}")

print(F)
print(X)
