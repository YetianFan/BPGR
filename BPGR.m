%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  BPGR algorithm
%%%%%%  BP algorithm with graph regularization
%%%%%%  Date 2021-12-13
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Name = 'iris';                    % name of datasets
HiddenUnitNum = 3;                % number of hidden nodes
MaxEpochs = 1000;                 % maximum number of iteration
b = 0.5;                          % inital weights [-b,b]
n_times = 50;                     % number of experiments
lr=0.003;                         %  learning rate
E0=0.1;                           %  the threshold of error 
lambda = 10.^(-5:1:0);                %  the set for lambda
n_crossvalind = 5;                %  the k-fold cross validation
n_function = 5;                   %  the number of activition function
a = 1;                            %  the threshold for ELU function

%%%%%%%%%%%%%%%%%%%%%%%%% loading data and process the data  %%%%%%%%%%%%%%%%%%%%%%%%%
FileTitle = [Name,'.txt'];
load(FileTitle);
data = eval(Name);
label_vector = data(:,1);
sample = data(:,2:end);
[SampleNum,n] = size(sample);
[label_vector,label_order]= sort(label_vector);
sample = sample(label_order,:);
InDim = n;                                             
range = max(sample)-min(sample);
label  = range==0;
range(label) = 1;
sample = 2*(sample - repmat(min(sample),SampleNum,1))./repmat(range,SampleNum,1) - 1 ;
sample = sample .* repmat(~label,SampleNum,1);
label = unique(label_vector);
number_class=length(label);
OutDim=number_class;
TargetOut = label_vector';
for i = 1:number_class
    TargetOut(label_vector == label(i)) = i;
end
temp_T=zeros( SampleNum,OutDim);                 
for i = 1:SampleNum
    temp_T(i,TargetOut(i))=1;
end
T = temp_T;
width = sumsqr(sample)/SampleNum;


for k = 1:n_times

    %%%%%%%%%%%%%%%%%%%%%%%%% divide training data into k parts  %%%%%%%%%%%%%%%%%%%%%%%%%
    indices = crossvalind('Kfold', SampleNum, n_crossvalind);
    for i = 1:n_crossvalind
        test = (indices == i);
        train = ~test;
        TestNum = sum(test);
        TrainNum = SampleNum - TestNum;
        TrainIn = sample(train,:)';
        TrainOut = T(train,:)';
        TestIn = sample(test,:)';
        TestOut = TargetOut(test);
		
		
		%%%%%%%%%%%%%%%%%%%%%%%%% generate the Laplacian matrix L  %%%%%%%%%%%%%%%%%%%%%%%%%
        rand('seed',sum(100*clock))
        num_class = [];
        for j = 1:number_class
            num_class(j) = sum(TargetOut(train) == j);
        end
        num_class = [0, num_class];
        num_class = cumsum(num_class);
        L = {};
        for j = 1:number_class
            data = TrainIn(:,num_class(j)+1:num_class(j+1));
            DD=pdist(data');
            D=squareform(DD);
            D = exp(-D.^2/width);
            LDiag = sum(D);
            Lap = diag(LDiag) - D;
            for l = 1:num_class(j+1)- num_class(j)
                dll = Lap(l,l);
                if dll>0
                    Lap(:,l) = Lap(:,l)./sqrt(dll);
                    Lap(l,:) = Lap(l,:)./sqrt(dll);
                end
            end
            L{j} = Lap;
        end
        
		
		%%%%%%%%%%%%%%%%%%%%%%%%%  generate the inital weights for neural networks  %%%%%%%%%%%%%%%%%%%%%%%%%
        W1=2*b*rand(HiddenUnitNum,InDim)-b;                
        B1=2*b*rand(HiddenUnitNum,1)-b;                    
        W2=2*b*rand(OutDim,HiddenUnitNum)-b;               
        B2=2*b*rand(OutDim,1)-b;                           
        
		%%%%%%%%%%%%%%%%%%%%%%%%%  Sigmoid activition function for hidden layer  %%%%%%%%%%%%%%%%%%%%%%%%%
        count_function = 1;
        [TrainNetworkOut,TestNetworkOut,ErrHistory] = BPLogsig(TrainIn,TrainOut,TestIn,W1,B1,W2,B2,HiddenUnitNum,MaxEpochs,lr,E0,L,num_class,0);
        [C,I] = max(TrainNetworkOut);
        I = I - 1 + min(TargetOut(train));
        TrAcc((i-1)*n_function+count_function,1) = sum(TargetOut(train)==I)/TrainNum*100;
        [C,I] = max(TestNetworkOut);
        I = I - 1 + min(TestOut);
        TeAcc((i-1)*n_function+count_function,1) = sum(TestOut==I)/TestNum*100;
        
        for j = 1:length(lambda)
            [TrainNetworkOut,TestNetworkOut,ErrHistory] = BPLogsig(TrainIn,TrainOut,TestIn,W1,B1,W2,B2,HiddenUnitNum,MaxEpochs,lr,E0,L,num_class,lambda(j));
            [C,I] = max(TrainNetworkOut);
            I = I - 1 + min(TargetOut(train));
            TrAcc((i-1)*n_function+count_function,j+1) = sum(TargetOut(train)==I)/TrainNum*100;
            [C,I] = max(TestNetworkOut);
            I = I - 1 + min(TestOut);
            TeAcc((i-1)*n_function+count_function,j+1) = sum(TestOut==I)/TestNum*100;
        end
        count_function = count_function + 1;
        
		%%%%%%%%%%%%%%%%%%%%%%%%%  TanH activition function for hidden layer %%%%%%%%%%%%%%%%%%%%%%%%%
        [TrainNetworkOut,TestNetworkOut,ErrHistory] = BPTansigSig(TrainIn,TrainOut,TestIn,W1,B1,W2,B2,HiddenUnitNum,MaxEpochs,lr,E0,L,num_class,0);
        [C,I] = max(TrainNetworkOut);
        I = I - 1 + min(TargetOut(train));
        TrAcc((i-1)*n_function+count_function,1) = sum(TargetOut(train)==I)/TrainNum*100;
        [C,I] = max(TestNetworkOut);
        I = I - 1 + min(TestOut);
        TeAcc((i-1)*n_function+count_function,1) = sum(TestOut==I)/TestNum*100;
        
        for j = 1:length(lambda)
            [TrainNetworkOut,TestNetworkOut,ErrHistory] = BPTansigSig(TrainIn,TrainOut,TestIn,W1,B1,W2,B2,HiddenUnitNum,MaxEpochs,lr,E0,L,num_class,lambda(j));
            [C,I] = max(TrainNetworkOut);
            I = I - 1 + min(TargetOut(train));
            TrAcc((i-1)*n_function+count_function,j+1) = sum(TargetOut(train)==I)/TrainNum*100;
            [C,I] = max(TestNetworkOut);
            I = I - 1 + min(TestOut);
            TeAcc((i-1)*n_function+count_function,j+1) = sum(TestOut==I)/TestNum*100;
        end
        count_function = count_function + 1;
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%  ReLU activition function for hidden layer  %%%%%%%%%%%%%%%%%%%%%%%%%
        [TrainNetworkOut,TestNetworkOut,ErrHistory] = BPReLUSig(TrainIn,TrainOut,TestIn,W1,B1,W2,B2,HiddenUnitNum,MaxEpochs,lr,E0,L,num_class,0);
        [C,I] = max(TrainNetworkOut);
        I = I - 1 + min(TargetOut(train));
        TrAcc((i-1)*n_function+count_function,1) = sum(TargetOut(train)==I)/TrainNum*100;
        [C,I] = max(TestNetworkOut);
        I = I - 1 + min(TestOut);
        TeAcc((i-1)*n_function+count_function,1) = sum(TestOut==I)/TestNum*100;
        
        for j = 1:length(lambda)
            [TrainNetworkOut,TestNetworkOut,ErrHistory] = BPReLUSig(TrainIn,TrainOut,TestIn,W1,B1,W2,B2,HiddenUnitNum,MaxEpochs,lr,E0,L,num_class,lambda(j));
            [C,I] = max(TrainNetworkOut);
            I = I - 1 + min(TargetOut(train));
            TrAcc((i-1)*n_function+count_function,j+1) = sum(TargetOut(train)==I)/TrainNum*100;
            [C,I] = max(TestNetworkOut);
            I = I - 1 + min(TestOut);
            TeAcc((i-1)*n_function+count_function,j+1) = sum(TestOut==I)/TestNum*100;
        end
        count_function = count_function + 1;
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%  ELU activition function for hidden layer  %%%%%%%%%%%%%%%%%%%%%%%%%
        [TrainNetworkOut,TestNetworkOut,ErrHistory] = BPELUSig(TrainIn,TrainOut,TestIn,W1,B1,W2,B2,HiddenUnitNum,MaxEpochs,a,lr,E0,L,num_class,0);
        [C,I] = max(TrainNetworkOut);
        I = I - 1 + min(TargetOut(train));
        TrAcc((i-1)*n_function+count_function,1) = sum(TargetOut(train)==I)/TrainNum*100;
        [C,I] = max(TestNetworkOut);
        I = I - 1 + min(TestOut);
        TeAcc((i-1)*n_function+count_function,1) = sum(TestOut==I)/TestNum*100;
        
        for j = 1:length(lambda)
            [TrainNetworkOut,TestNetworkOut,ErrHistory] = BPELUSig(TrainIn,TrainOut,TestIn,W1,B1,W2,B2,HiddenUnitNum,MaxEpochs,a,lr,E0,L,num_class,lambda(j));
            [C,I] = max(TrainNetworkOut);
            I = I - 1 + min(TargetOut(train));
            TrAcc((i-1)*n_function+count_function,j+1) = sum(TargetOut(train)==I)/TrainNum*100;
            [C,I] = max(TestNetworkOut);
            I = I - 1 + min(TestOut);
            TeAcc((i-1)*n_function+count_function,j+1) = sum(TestOut==I)/TestNum*100;
        end
        count_function = count_function + 1;
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%  GELU activition function for hidden layer   %%%%%%%%%%%%%%%%%%%%%%%%%
        [TrainNetworkOut,TestNetworkOut,ErrHistory] = BPGELUSig(TrainIn,TrainOut,TestIn,W1,B1,W2,B2,HiddenUnitNum,MaxEpochs,lr,E0,L,num_class,0);
        [C,I] = max(TrainNetworkOut);
        I = I - 1 + min(TargetOut(train));
        TrAcc((i-1)*n_function+count_function,1) = sum(TargetOut(train)==I)/TrainNum*100;
        [C,I] = max(TestNetworkOut);
        I = I - 1 + min(TestOut);
        TeAcc((i-1)*n_function+count_function,1) = sum(TestOut==I)/TestNum*100;
        
        for j = 1:length(lambda)
            [TrainNetworkOut,TestNetworkOut,ErrHistory] = BPGELUSig(TrainIn,TrainOut,TestIn,W1,B1,W2,B2,HiddenUnitNum,MaxEpochs,lr,E0,L,num_class,lambda(j));
            [C,I] = max(TrainNetworkOut);
            I = I - 1 + min(TargetOut(train));
            TrAcc((i-1)*n_function+count_function,j+1) = sum(TargetOut(train)==I)/TrainNum*100;
            [C,I] = max(TestNetworkOut);
            I = I - 1 + min(TestOut);
            TeAcc((i-1)*n_function+count_function,j+1) = sum(TestOut==I)/TestNum*100;
        end
        count_function = count_function + 1;   
    end
    
 
    for j = 1:n_function
        id = (1:n_function:n_function*n_crossvalind) + j - 1;
        Acc = TrAcc(id,:);
        TrainAcc((k-1)*n_function+j,:) = mean(Acc);
        Acc = TeAcc(id,:);
        TestAcc((k-1)*n_function+j,:) = mean(Acc);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%% calculate the average performance   %%%%%%%%%%%%%%%%%%%%%%%%%
if n_times > 1
    for i = 1:n_function
        id = (1:n_function:n_function*n_times) + i - 1;
        Acc = TrainAcc(id,:);
        TrainAccuracy(i,:) = mean(Acc);
        Acc = TestAcc(id,:);
        TestAccuracy(i,:) = mean(Acc);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%% write the results   %%%%%%%%%%%%%%%%%%%%%%%%%
FileTitle = [Name,'BPResult.txt'];
fid = fopen(FileTitle,'wt');
fprintf(fid,'TrainAcc \n');
[m,n] = size(TrainAcc);
for i = 1:m
    for j = 1:n
        fprintf(fid,'%6.2f\t',TrainAcc(i,j));
    end
    fprintf(fid,'\n');
end
fprintf(fid,'\n');

fprintf(fid,'TestAcc \n');
[m,n] = size(TestAcc);
for i = 1:m
    for j = 1:n
        fprintf(fid,'%6.2f\t',TestAcc(i,j));
    end
    fprintf(fid,'\n');
end
fprintf(fid,'\n');

fprintf(fid,'TrainAccuracy \n');
[m,n] = size(TrainAccuracy);
for i = 1:m
    for j = 1:n
        fprintf(fid,'%6.2f\t',TrainAccuracy(i,j));
    end
    fprintf(fid,'\n');
end
fprintf(fid,'\n');

fprintf(fid,'TestAccuracy \n');
[m,n] = size(TestAccuracy);
for i = 1:m
    for j = 1:n
        fprintf(fid,'%6.2f\t',TestAccuracy(i,j));
    end
    fprintf(fid,'\n');
end
fprintf(fid,'\n');

fclose(fid);




