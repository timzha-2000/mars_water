function out = myBerryFixedSw(theta, H, fixed_sw)
% Identical to myBerry but with S_w hardcoded to fixed_sw (binary mode).
asp = [1 theta(1)];
x_phi = theta(2);
rock_vol = 1-theta(2);
x = [rock_vol, x_phi];
rock_density = theta(6).* rock_vol;
gas_density = 0.020.* x_phi;
rhob1 = rock_density + gas_density;
P_water = fixed_sw;
k  = [theta(4)*1e9 0];
mu = [theta(5)*1e9 0];

[~,~,vp,vs,rhob,~] = berryscm(k,mu,asp,x,rhob1,P_water);

if sum(H) == 3
    out = [[vp; vs]./1e3; rhob];
elseif sum(H)==2
    if H(1)==1 && H(2) == 1
         out = [vp; vs]./1e3;
    elseif H(1)==1 && H(3) == 1
        out = [vp/1e3; rhob];
    elseif H(2)==1 && H(3) == 1
        out = [vs/1e3; rhob];
    else
        error('What?')
    end
elseif sum(H)==1
    if H(1)==1
        out = vp/1e3;
    elseif H(2)==1
        out = vs/1e3;
    elseif H(3)==1
        out = rhob;
    end
end
end
