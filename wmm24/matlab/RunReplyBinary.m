%% RunReplyBinary.m -- Berryman+Voigt, Vp=4.1, Wright 2025 Reply priors, BINARY S_w
clearvars; close all; clc
rng(20240605, 'twister');

outdir = fullfile(pwd, 'outputs_appendix_priors');
if ~exist(outdir, 'dir'), mkdir(outdir); end

lb = [0.01  0.01  0  35  20  2600]';
ub = [0.99  0.40  1  80  50  3100]';
n  = length(ub);

d = [4.1; 2.5; 2589];  s = [0.2; 0.3; 157];
H = [1;1;1];

%% WET branch
fixed_sw_wet = 1.0;
logpi_wet = @(x) myLogPiFixedSw(x, lb, ub, d, s, H, fixed_sw_wet);

Ne = 3*n;
Xo = zeros(n, Ne);
counter = 0; tries = 0;
while counter < Ne && tries < 1e6
    tries = tries + 1;
    xo = lb + (ub-lb).*rand(n,1);
    xo(3) = fixed_sw_wet;
    if isfinite(myLogPiFixedSw(xo, lb, ub, d, s, H, fixed_sw_wet))
        counter = counter + 1;
        Xo(:,counter) = xo;
    end
end
fprintf('[wet] init %d walkers after %d tries\n', counter, tries);

[X, ~, LogPi, ~] = myHammer(500, Xo, 2.6, logpi_wet, H);
Xrs = reshape(X, [n, size(X,2)*size(X,3)]);
LogPirs = reshape(LogPi, [1, numel(LogPi)]);
RMSE = sqrt(2*LogPirs/length(d));
Xrs = Xrs(:, RMSE < 3);
Xo  = Xrs(:, randi(length(Xrs), Ne, 1));

tic
[X, D, LogPi, AccRatio] = myHammer(5e4, Xo, 2.6, logpi_wet, H);
fprintf('[wet] warm chain %.1f s, acc.ratio %.3f\n', toc, AccRatio);

X     = X(:, 1e3:end, :);
LogPi = LogPi(:, 1e3:end);
Xrs   = reshape(X, [n, size(X,2)*size(X,3)]);
thickness_wet = 8500 * fixed_sw_wet * Xrs(2,:);

%% Prior-MC evidence
N_evidence = 100000;
fprintf('Prior-MC evidence: N=%d ...\n', N_evidence);
Xprior = lb + (ub - lb) .* rand(n, N_evidence);
logL_wet = -inf(1, N_evidence);
logL_dry = -inf(1, N_evidence);
for i = 1:N_evidence
    xi = Xprior(:,i);
    logL_wet(i) = myLogPiFixedSw(xi, lb, ub, d, s, H, 1.0);
    logL_dry(i) = myLogPiFixedSw(xi, lb, ub, d, s, H, 0.0);
end
mw = max(logL_wet); md = max(logL_dry);
logZ_wet = mw + log(mean(exp(logL_wet - mw)));
logZ_dry = md + log(mean(exp(logL_dry - md)));
P_wet = 1.0 / (1.0 + exp(logZ_dry - logZ_wet));
fprintf('logZ_wet=%.3f, logZ_dry=%.3f, P(wet)=%.4f\n', logZ_wet, logZ_dry, P_wet);

writeNPY(Xrs.', fullfile(outdir, 'samples_reply_binary_wet.npy'));
writeNPY(thickness_wet(:), fullfile(outdir, 'thickness_samples_reply_binary_wet.npy'));
fid = fopen(fullfile(outdir, 'logZ_reply_wet.txt'),'w'); fprintf(fid, '%.6f\n', logZ_wet); fclose(fid);
fid = fopen(fullfile(outdir, 'logZ_reply_dry.txt'),'w'); fprintf(fid, '%.6f\n', logZ_dry); fclose(fid);

NMIX = 200000;
n_wet = round(P_wet * NMIX);
mix = [datasample(thickness_wet, n_wet, 'Replace', true), zeros(1, NMIX - n_wet)];
fprintf('Binary mixture: P(wet)=%.3f, median %.1f m, mean %.1f m, 95%% %.1f m\n', ...
        P_wet, median(mix), mean(mix), prctile(mix, 95));

fid = fopen(fullfile(outdir, 'summary_reply_binary.txt'),'w');
fprintf(fid, 'Set: Reply, binary S_w, Vp=4.1\n');
fprintf(fid, 'P(wet)=%.4f, logZ_wet=%.3f, logZ_dry=%.3f\n', P_wet, logZ_wet, logZ_dry);
fprintf(fid, 'mix: median=%.1f m, mean=%.1f m, 5%%=%.1f m, 95%%=%.1f m\n', ...
        median(mix), mean(mix), prctile(mix,5), prctile(mix,95));
fclose(fid);

fprintf('DONE reply_binary\n');
exit(0)
