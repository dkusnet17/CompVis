function P=camcalibDLT(Xworld,Xim)

N=size(Xworld,1);

A=[];
for i=1:N
    A=[A;...
       zeros(1,4) Xworld(i,:) -Xim(i,2)*Xworld(i,:);...
       Xworld(i,:) zeros(1,4) -Xim(i,1)*Xworld(i,:)];
end

M=A'*A;

[V,D]=eig(M);
[emin,idmin]=min(diag(D));
ev=V(:,idmin);

P=(reshape(ev,4,3))';

%keyboard
end