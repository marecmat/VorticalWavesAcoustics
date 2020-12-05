clc;clear all;close all;



clear all;
param=init_parametre;
f=1000;
omega=2.*pi.*f;
k=omega./param.c_0;


% half the distance between the two rigid walls
d=5e-2;

%number of reflections taken into account in the definition of the image sources
N=1000; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% position of all sources %%%%%%%%%%%%%%%%%%
% (primary plus all image sources of order lower than N) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_source=zeros(1,2.*N+1);
y_source=linspace(-2.*N.*d,2.*N.*d,2.*N+1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% position of the observation points (x,y) %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=20.*d;
y=linspace(-d,d,51);




for kk=1:length(y),

for ii=1:(2.*N+1),
    r(ii)=sqrt((x-x_source(ii)).^2+(y(kk)-y_source(ii)).^2);
    [p_tilde(ii)]=calcul_champ_pulsating_string(f,param,r(ii));
end

p_tilde_tot(kk)=sum(p_tilde);
end

figure(1);
subplot(211);plot(y./d,abs(p_tilde_tot),'r','LineWidth',2);
xlabel('y/d','FontSize',20);ylabel('|p| (Pa)','FontSize',20);ylim([0 1.1.*max(abs(p_tilde_tot))]);grid on;
subplot(212);plot(y./d,unwrap(angle(p_tilde_tot)),'r','LineWidth',2);
xlabel('y/d','FontSize',20);ylabel('Arg(p) (rad)','FontSize',20);ylim([-pi pi]);grid on;

critere_coupure=f./(param.c_0./(2.*(2.*d)))


