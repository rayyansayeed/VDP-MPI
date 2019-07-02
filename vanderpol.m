function p = vanderpol(alpha,sigma,M,N)
% p = vanderpol(alpha,sigma,M,N)
% simulate Van der Pol oscillator with noise
% alpha = location of sinks at +/-alpha,0
% sigma = strength of noise
% M = number of time steps
% N = number of trials

T = 10.;
dt=T/M;
t = 0:dt:T;
k = [1,M/10:M/10:M];
p = zeros(size(k));
z=zeros(N,2);
for trial=1:N
    x = zeros(2,length(t));
    x(:,1) = [0.1*randn;0];
    for j=2:length(t)
        x(1,j) = x(1,j-1) + x(2,j-1)*dt;
        x(2,j) = x(2,j-1) + ((alpha^2-x(1,j-1)^2)*x(1,j-1)-x(2,j-1))*dt + sigma*x(1,j-1)*sqrt(dt)*randn;
    end
    X = x(1,k);
    Y = x(2,k);
    d1 = sqrt((X-alpha).^2+Y.^2) <= alpha/2;
    d2 = sqrt((X+alpha).^2+Y.^2) <= alpha/2;
    p = p+d1+d2;
    plot([0,k(2:end)]*dt,p/trial,'bo-');
    axis([0,T,0,1]);
    drawnow
end
p = p/trial;
end