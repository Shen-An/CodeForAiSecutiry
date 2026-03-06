% loss_bpf.m
% 读取 results/loss_block_per_mean.csv 并画损失曲面：dx, dy -> loss_mean

clear; clc;

csvPath = fullfile('results', 'loss_block_per_mean (3).csv');

T = readtable(csvPath);

needCols = {'dx','dy','loss_mean'};
for k = 1:numel(needCols)
    if ~ismember(needCols{k}, T.Properties.VariableNames)
        error("CSV 缺少列: %s", needCols{k});
    end
end

dx = T.dx;
dy = T.dy;
loss = T.loss_mean;

% ---- 网格 ----
xv = unique(dx, 'sorted');
yv = unique(dy, 'sorted');
[X, Y] = meshgrid(xv, yv);

% ---- 填 Z ----
[~, ix] = ismember(dx, xv);
[~, iy] = ismember(dy, yv);
Z = nan(numel(yv), numel(xv));
idx = sub2ind(size(Z), iy, ix);
Z(idx) = loss;

% 如有缺失点，插值补齐（避免 surf/contourf 出现白色空洞）
% 注意：你的 CSV 里只有 (0,0) + (0.1..0.5)x(0.1..0.5)，边界上会缺点，Z 会出现 NaN。
if any(isnan(Z(:)))
    fprintf("Warning: Z 中存在 NaN，使用 scatteredInterpolant 补齐缺失点...\n");
    % 'natural' 在内部插值较平滑；外推用 'nearest' 防止边界仍为 NaN
    Fi = scatteredInterpolant(dx, dy, loss, 'natural', 'nearest');
    Zi = Fi(X, Y);
    Z(isnan(Z)) = Zi(isnan(Z));
end

outDir = 'results';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% ---- 3D 曲面 ----
figure('Color','w');
% 避免部分显卡/驱动下 OpenGL + lighting 的白色渲染伪影：先强制固定渲染器
set(gcf, 'Renderer', 'opengl');

surf(X, Y, Z, 'EdgeColor', 'none');
colormap(parula);
colorbar;
xlabel('\rho');
ylabel('\rho');
zlabel('mean cross-entropy loss',FontName="Times New Roman");
title('Loss Landscape (GIA)',FontName="Times New Roman");
view(-45, 35);
grid on;

% 光照在某些机器上会触发“移动才正常”的显示伪影；这里默认关闭。
% 如果你确实需要光照效果，可以把下面两行取消注释。
% camlight headlight; lighting gouraud;

% ---- 等高线 ----
figure('Color','w');
set(gcf, 'Renderer', 'opengl');

contourf(X, Y, Z, 30, 'LineColor', 'none');
colormap(parula);
colorbar;
xlabel('\rho');
ylabel('\rho');
title('Loss Contour (GIA)');
axis image;

% ---- 保存 ----
figSurf = figure(1);
outPngSurf = fullfile(outDir, 'loss_blockper_surface.png');
outEpsSurf = fullfile(outDir, 'loss_blockper_surface.eps');
outFigSurf = fullfile(outDir, 'loss_blockper_surface.fig');
saveas(figSurf, outPngSurf);
saveas(figSurf, outFigSurf);
print(figSurf, outEpsSurf, '-depsc', '-painters');

figContour = figure(2);
outPngContour = fullfile(outDir, 'loss_blockper_contour.png');
outEpsContour = fullfile(outDir, 'loss_blockper_contour.eps');
outFigContour = fullfile(outDir, 'loss_blockper_contour.fig');
saveas(figContour, outPngContour);
saveas(figContour, outFigContour);
print(figContour, outEpsContour, '-depsc', '-painters');

fprintf("Saved:\n  %s\n  %s\n  %s\n  %s\n", outPngSurf, outEpsSurf, outPngContour, outEpsContour);
