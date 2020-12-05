function [p_tilde]=calcul_champ_pulsating_string(f,param,r);

w=2.*pi.*f;
k=w./param.c_0;
p_tilde=i.*param.rho_0.*w.*param.V_0./besselh(1,2,k.*param.a).*besselh(0,2,k.*r);

%p_tilde=besselh(0,2,k.*r);