%% RunReplyCont.m -- Berryman+Voigt, Vp=4.1, Wright 2025 Reply priors, continuous S_w
clearvars; close all; clc
rng(20240603, 'twister');

outdir = fullfile(pwd, 'outputs_appendix_priors');
if ~exist(outdir, 'dir'), mkdir(outdir); end

% Wright 2025 Reply bounds
lb = [0.01  0.01  0  35  20  2600]';
ub = [0.99  0.40  1  80  50  3100]';
n  = length(ub);

d = [4.1; 2.5; 2589];  s = [0.2; 0.3; 157];
H = [1;1;1];

logpi = @(x) myLogPi(x, lb, ub, d, s, H);

Ne = 3*n;
Xo = zeros(n, Ne);
counter = 0; tries = 0;
while counter < Ne && tries < 1e6
    tries = tries + 1;
    xo = lb + (ub-lb).*rand(n,1);
    if isfinite(myLogPi(xo, lb, ub, d, s, H))
        counter = counter + 1;
        Xo(:,counter) = xo;
    end
end
fprintf('Initialized %d walkers after %d tries\n', counter, tries);

[X, ~, LogPi, ~] = myHammer(500, Xo, 2.6, logpi, H);
Xrs = reshape(X, [n, size(X,2)*size(X,3)]);
LogPirs = reshape(LogPi, [1, numel(LogPi)]);
RMSE = sqrt(2*LogPirs/length(d));
Xrs = Xrs(:, RMSE < 3);
Xo  = Xrs(:, randi(length(Xrs), Ne, 1));

tic
[X, D, LogPi, AccRatio] = myHammer(5e4, Xo, 2.6, logpi, H);
fprintf('Warm chain done in %.1f s, acc.ratio %.3f\n', toc, AccRatio);

X     = X(:, 1e3:end, :);
LogPi = LogPi(:, 1e3:end);
Xrs   = reshape(X, [n, size(X,2)*size(X,3)]);

thickness = 8500 * Xrs(3,:) .* Xrs(2,:);
fprintf('Mean thickness %.1f m, median %.1f m, 95%% %.1f m\n', ...
        mean(thickness), median(thickness), prctile(thickness,95));

writeNPY(Xrs.', fullfile(outdir, 'samples_reply_cont.npy'));
writeNPY(thickness(:), fullfile(outdir, 'thickness_samples_reply_cont.npy'));

fid = fopen(fullfile(outdir, 'summary_reply_cont.txt'), 'w');
fprintf(fid, 'Set: Reply (Wright 2025), continuous S_w, Vp=4.1\n');
fprintf(fid, 'mean=%.1f m, median=%.1f m, 5%%=%.1f m, 95%%=%.1f m, acc=%.3f\n', ...
        mean(thickness), median(thickness), prctile(thickness,5), prctile(thickness,95), AccRatio);
fclose(fid);

fprintf('DONE reply_cont\n');
exit(0)
