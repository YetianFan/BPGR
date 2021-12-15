function value = dGELU(x)
tanhvalue = tanh(sqrt(2/pi)*(x+0.044715*x.^3));
value = 0.5*(1+tanhvalue) +  0.5*x.*(1-tanhvalue.^2)*sqrt(2/pi).*(1+0.044715*3*x.^2);