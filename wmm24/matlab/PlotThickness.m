%% PlotThickness.m
% Load the MATLAB inversion results for cases A/B/C and plot the
% water-layer thickness posterior (native MATLAB), with summary stats.
% Toolbox-free: uses a local Gaussian KDE and percentile helper so it
% runs without the Statistics and Machine Learning Toolbox.
clearvars; close all; clc
outdir = fullfile(pwd, 'outputs_bm_matlab');

cases  = {'A_constraints_away', 'B_wright_inherited', 'C_insight_marsquake'};
labels = {'A: V_p=3.8 km/s', 'B: V_p=4.1 km/s', 'C: V_p=4.7 km/s'};
cols   = [0 0.45 0.70; 0.90 0.62 0; 0.6 0.6 0.6];   % blue / orange / grey

WRIGHT_MODE_KM = 1.248;   % Wright et al. (2024) peak-based mode reference

figure('Color','w','Position',[100 100 900 560]); hold on
xq = linspace(0, 4, 600);   % thickness in km
fprintf('\n%-22s   mean   median     5%%     95%%   (km)\n', 'case');
ymax = 0; hLine = gobjects(1, numel(cases));
for ci = 1:numel(cases)
    S = load(fullfile(outdir, sprintf('results_%s.mat', cases{ci})), 'thickness');
    th_km = S.thickness(:) / 1000;                % m -> km
    f = gaussKDE(th_km, xq);                        % kernel density estimate
    fill([xq fliplr(xq)], [f zeros(size(f))], cols(ci,:), ...
         'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility','off');
    hLine(ci) = plot(xq, f, 'Color', cols(ci,:), 'LineWidth', 2.5);
    ymax = max(ymax, max(f));
    pc = pctl(th_km, [5 50 95]);
    fprintf('%-22s  %6.3f  %6.3f  %6.3f  %6.3f\n', ...
            cases{ci}, mean(th_km), pc(2), pc(1), pc(3));
end
hMode = plot(WRIGHT_MODE_KM, 0, 'p', 'MarkerSize', 18, 'MarkerFaceColor','r', ...
     'MarkerEdgeColor','k', 'LineWidth', 0.8);
xlabel('Water-layer thickness [km]', 'FontSize', 16)
ylabel('Probability density', 'FontSize', 16)
title('Thickness posterior (WMM24 Berryman SC, MATLAB)', 'FontSize', 14)
legend([hLine, hMode], [labels, {'Wright et al. (2024) mode'}], ...
       'Location','northeast', 'FontSize', 13, 'Box','off')
set(gca, 'FontSize', 13); box off; xlim([0 4]); ylim([0 ymax*1.05])

exportgraphics(gcf, fullfile(outdir, 'thickness_posterior_matlab.png'), 'Resolution', 200);
exportgraphics(gcf, fullfile(outdir, 'thickness_posterior_matlab.pdf'));
fprintf('\nSaved thickness_posterior_matlab.png / .pdf to %s\n', outdir);

%% ---- local helpers (toolbox-free) -------------------------------------
function f = gaussKDE(x, xq)
    x = x(:); n = numel(x);
    s = std(x); iqr = diff(pctl(x, [25 75]));
    sig = min(s, iqr/1.349); if sig <= 0, sig = s; end
    h = 0.9 * sig * n^(-1/5);              % Silverman's rule of thumb
    f = zeros(size(xq));
    for k = 1:numel(xq)
        u = (xq(k) - x) / h;
        f(k) = sum(exp(-0.5*u.^2)) / (n*h*sqrt(2*pi));
    end
end

function q = pctl(x, p)
    x = sort(x(:)); n = numel(x);
    % linear interpolation on the (k-0.5)/n plotting positions
    pos = ((1:n) - 0.5) / n * 100;
    q = interp1(pos, x, p, 'linear', NaN);
    q(p <= pos(1))   = x(1);
    q(p >= pos(end)) = x(end);
end
