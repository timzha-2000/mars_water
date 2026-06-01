%% RunAllCases.m
% Re-run the WMM24 Berryman self-consistent inversion (theory 1:
% Berryman + Voigt) for the three canonical cases A/B/C, using the
% ORIGINAL MATLAB code from mattimorzfeld/WMM24, and compute the
% water-layer thickness posterior natively in MATLAB.
%
% Thickness transform (from Berryman_mod.ipynb):
%     thickness_m = 8500 * water_saturation * porosity
%                 = 8500 * Xrs(3,:) .* Xrs(2,:)
%
% Parameter rows of Xrs: [asp; phi; water; k_GPa; mu_GPa; rho_min]
clearvars; close all; clc
rng(20240501, 'twister');   % reproducible

outdir = fullfile(pwd, 'outputs_bm_matlab');
if ~exist(outdir, 'dir'), mkdir(outdir); end

% Canonical cases (Vp, Vs, rho) and noise std (sigVp, sigVs, sigRho)
% -- identical to bm_github/run.py used across all theories.
cases = struct( ...
    'name', {'A_constraints_away', 'B_wright_inherited', 'C_insight_marsquake'}, ...
    'd',    {[3.8; 2.2; 2589],     [4.1; 2.5; 2589],      [4.7; 2.7; 2589]}, ...
    's',    {[1.0; 0.4; 157],      [0.2; 0.3; 157],       [0.3; 0.1; 157]});

% Parameter bounds: asp phi water k mu rho_min
lb = [0.0  0.0   0  75.6 25.6 2680]';
ub = [1    0.50  1  80   40   2900]';
n  = length(ub);

COL_STRETCH = 8500;   % saturated-column thickness scale [m]

summary = struct();
for ci = 1:numel(cases)
    name = cases(ci).name;
    d = cases(ci).d;  s = cases(ci).s;
    H = [1;1;1];  nData = sum(H);
    fprintf('\n==================== CASE %s ====================\n', name);
    fprintf('d = [%g %g %g], s = [%g %g %g]\n', d(1),d(2),d(3),s(1),s(2),s(3));

    logpi = @(x) myLogPi(x, lb, ub, d, s, H);

    %% Initialize ensemble with parameters satisfying all constraints
    Ne = 3*n;
    Xo = zeros(n, Ne);
    counter = 0;
    while counter < Ne
        xo = lb + (ub-lb).*rand(n,1);
        if isfinite(myLogPi(xo, lb, ub, d, s, H))
            counter = counter + 1;
            Xo(:,counter) = xo;
        end
    end

    %% Cold start
    Nsteps = 5e2;
    [X, ~, LogPi, ~] = myHammer(Nsteps, Xo, 2.6, logpi, H);
    Xrs = reshape(X, [n, size(X,2)*size(X,3)]);
    LogPirs = reshape(LogPi, [1, numel(LogPi)]);
    RMSE = sqrt(2*LogPirs/length(d));
    Xrs = Xrs(:, RMSE < 3);
    Xo  = Xrs(:, randi(length(Xrs), Ne, 1));

    %% Warm start (production chain)
    Nsteps = 5e4;
    tic
    [X, D, LogPi, AccRatio] = myHammer(Nsteps, Xo, 2.6, logpi, H);
    fprintf('\n[%s] warm chain done in %.1f s, acc. ratio = %.3f\n', name, toc, AccRatio);

    %% Burn-in + flatten
    BurnIn = 1e3;
    X     = X(:, BurnIn:end, :);
    D     = D(:, BurnIn:end, :);
    LogPi = LogPi(:, BurnIn:end);
    Xrs     = reshape(X, [n, size(X,2)*size(X,3)]);
    Drs     = reshape(D, [nData, size(D,2)*size(D,3)]);
    LogPirs = reshape(LogPi, [1, numel(LogPi)]);
    RMSE = sqrt(2*LogPirs/length(d));

    %% Thickness posterior: 8500 * water * porosity
    thickness = COL_STRETCH * Xrs(3,:) .* Xrs(2,:);   % [m], row vector

    %% Stats
    m  = mean(Xrs, 2);  sp = std(Xrs, [], 2);
    pc = prctile(thickness, [5 50 95]);
    fprintf('  Porosity phi : %.4f +/- %.4f\n', m(2), sp(2));
    fprintf('  Water gamma_w: %.4f +/- %.4f\n', m(3), sp(3));
    fprintf('  Thickness [m]: mean %.1f, median %.1f, 5%% %.1f, 95%% %.1f\n', ...
            mean(thickness), pc(2), pc(1), pc(3));

    %% Save MATLAB-native results
    matfile = fullfile(outdir, sprintf('results_%s.mat', name));
    save(matfile, 'Xrs', 'Drs', 'RMSE', 'thickness', 'd', 's', 'H', 'lb', 'ub', 'AccRatio');

    %% Export .npy (samples (Nsamp x 6) + thickness (Nsamp,)) for the
    %% cross-theory comparison pipeline (matches emcee flat schema).
    writeNPY(Xrs.', fullfile(outdir, sprintf('samples_%s.npy', name)));
    writeNPY(thickness(:), fullfile(outdir, sprintf('thickness_samples_%s.npy', name)));

    summary(ci).name = name;
    summary(ci).thickness_mean = mean(thickness);
    summary(ci).thickness_median = pc(2);
    summary(ci).thickness_p5 = pc(1);
    summary(ci).thickness_p95 = pc(3);
    summary(ci).nsamples = numel(thickness);
end

save(fullfile(outdir, 'summary.mat'), 'summary');
fprintf('\nAll cases done. Outputs in %s\n', outdir);
