function value = dELU(a,x)
value = (x>=0) + a.*exp(x).*(x<0); 