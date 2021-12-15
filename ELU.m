function value = ELU(a,x)
value = x .* (x>=0) + a.*(exp(x)-1).*(x<0); 