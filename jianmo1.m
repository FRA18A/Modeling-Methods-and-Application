clc
clear all;
s=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22];
d=[40 38 35 36 37 37 37 38 37 37 43 34 26 35 34 35 31 40 40 41 38 39 ];
p111=[d]; %输入数据矩阵
t111=[s]; %目标数据矩阵
[p,minp,maxp,t,mint,maxt]=premnmx(p111,t111); % 对于输入矩阵p111和输出矩阵t111进行归一化处理
dx=[-1,1;-1,1;-1,1];
% 定义训练样本
% P 为输入矢量
P=p;
% T 为目标矢量
T=t;
size(P);
size(T);
at_1=newff(minmax(P),[10,1],{'tansig','purelin'},'traingdm')%后面的只是必须添加
%at_1=newff(minmax(P),[3,8,1])%后面的只是必须添加
inputWeights=at_1.IW{1,1};
inputbias=at_1.b{1};
layerWeights=at_1.LW{2,1};
layerbias=at_1.b{2};
% 设置训练参数
at_1.trainParam.show = 500;
at_1.trainParam.lr = 0.1;
at_1.trainParam.mc = 0.9;
at_1.trainParam.epochs = 25000;
at_1.trainParam.goal = 1e-3;
[at_1,tr]=train(at_1,P,T);
A = sim(at_1,P)
an=postmnmx(A,mint,maxt)+23;
E = T - A;
MSE=mse(E)
x=1:22
plot(x,an,'r-o',x,d,'g--+')
%plot(x,an,'ro',x,d,'b-+')
legend('载客次数预测值','载客次数真实值')