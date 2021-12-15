function [TrainNetworkOut,TestNetworkOut,ErrHistory] = BPELUSig(TrainIn,TrainOut,TestIn,W1,B1,W2,B2,HiddenUnitNum,MaxEpochs,a,lr,E0,L,num_class,pL)

InDim = size(TrainIn,1);                         % dimension of input data
TrainNum = size(TrainIn,2);                      % number of training data
TestNum = size(TestIn,2);                        % number of test data
ErrHistory=[];                                   %  record the error

W1Ex=[W1  B1];                                      
W2Ex=[W2  B2];                                       
TrainInEx=[TrainIn' ones(TrainNum,1)]';                   

%%%%%%%%%%%%%%%%%%%%%%%%% the parameter for RMSprop approach  %%%%%%%%%%%%%%%%%%%%%%%%%
momentW1 = W1Ex*0;
momentW2 = W2Ex*0;
mu = 0.9;
epsilon = 10^(-8);

%%%%%%%%%%%%%%%%%%%%%%%%% train the neural network  %%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:MaxEpochs
    InToHidden=W1Ex*TrainInEx;
    HiddenOut=ELU(a,InToHidden);
    HiddenOutEx=[HiddenOut' ones(TrainNum, 1)]';
    NetworkOut=logsig(W2Ex*HiddenOutEx);
    
    Error=TrainOut-NetworkOut;
    SSE=sumsqr(Error);
    
    ErrHistory=[ErrHistory SSE];
    
    if SSE<E0, break, end
    
    Delta2=Error.*NetworkOut.*(1-NetworkOut);
    Delta1=W2'*Delta2.*dELU(a,InToHidden);
   
    dW2Ex=Delta2*HiddenOutEx';
    dW1Ex=Delta1*TrainInEx';
    
    if pL ~= 0
        HiddenIn  = W1 * TrainIn;
        for j = 1:length(num_class)-1
            HiddenLaplace(:,num_class(j)+1:num_class(j+1)) = HiddenIn(:,num_class(j)+1:num_class(j+1))*L{j};
        end
        dW1Ex(:,1:InDim) = dW1Ex(:,1:InDim) - HiddenLaplace * TrainIn' .* pL;
    end
    
    if i>1
        momentW1 = mu* momentW1 + (1-mu)*(dW1Ex.*dW1Ex);
        momentW2 = mu* momentW2 + (1-mu)*(dW2Ex.*dW2Ex);
        
        W1Ex=W1Ex+lr./(sqrt(momentW1)+epsilon).*dW1Ex;
        W2Ex=W2Ex+lr./(sqrt(momentW2)+epsilon).*dW2Ex;
    else
        W1Ex=W1Ex+lr*dW1Ex;
        W2Ex=W2Ex+lr*dW2Ex;
        momentW1 = dW1Ex.*dW1Ex;
        momentW2 = dW2Ex.*dW2Ex;
    end
    
    W1=W1Ex(:,1:InDim);
    W2=W2Ex(:,1:HiddenUnitNum);    
end

W1=W1Ex(:,1:InDim);
B1=W1Ex(:,InDim+1);
B2=W2Ex(:,1+HiddenUnitNum);

%%%%%%%%%%%%%%%%%%%%%%%%% calculate the training error  %%%%%%%%%%%%%%%%%%%%%%%%%
TrainInToHidden = W1*TrainIn+repmat(B1,1,TrainNum);
TrainHiddenOut=ELU(a,TrainInToHidden);
TrainNetworkOut=logsig(W2*TrainHiddenOut+repmat(B2,1,TrainNum));

%%%%%%%%%%%%%%%%%%%%%%%%% calculate the test error  %%%%%%%%%%%%%%%%%%%%%%%%%
TestInToHidden = W1*TestIn+repmat(B1,1,TestNum);
TestHiddenOut=ELU(a,TestInToHidden);
TestNetworkOut=logsig(W2*TestHiddenOut+repmat(B2,1,TestNum));
