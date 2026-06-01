function [logpi, dM] = myLogPiFixedSw(x, lb, ub, d, s, H, fixed_sw)
n = length(x);
dM = myBerryFixedSw(x, H, fixed_sw);
if sum(x>lb)==n && sum(x<=ub)==n
    if sum(H)==2 && H(1)==1 && H(2)==1
        if dM(1)>dM(2)
            logpi = -0.5*norm(s.\(d-dM)).^2;
        else
            logpi = -inf;
        end
    elseif sum(H)==3
        if dM(1)>dM(2)
            logpi = -0.5*norm(s.\(d-dM)).^2;
        else
            logpi = -inf;
        end
    elseif sum(H)==2 && H(1)==1 || H(2)==1
        if dM(2) < 3100 && dM(2) > 2500
            logpi = -0.5*norm(s.\(d-dM)).^2;
        else
            logpi = -inf;
        end
    elseif sum(H)==1 && H(3)==1
        if dM < 3100 && dM > 2500
            logpi = -0.5*norm(s.\(d-dM)).^2;
        else
            logpi = -inf;
        end
    else
        logpi = -0.5*norm(s.\(d-dM)).^2;
    end
else
    logpi = -inf;
end
end
