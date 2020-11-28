function param=init_parametre;

param.a=1e-3; %string's radius
param.V_0=1e-2;
param.P_0=1.013e5;
param.T_0=293;
param.R_gp=8.31;
param.M_mol=29e-3;
param.rho_0=param.P_0./(param.R_gp./param.M_mol.*param.T_0);
gamma=1.4;
param.c_0=sqrt(gamma.*param.P_0./param.rho_0);
