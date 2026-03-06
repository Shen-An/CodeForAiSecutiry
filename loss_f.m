% plot_loss_surface.m
% 读取 loss_perspective_mean.csv 并画损失曲面：dx, dy -> loss_mean

clear; clc;

% ---- 1) 路径（按你的工程结构）----
csvPath = fullfile('results', 'loss_block_pf_mean.csv');
outPng  = fullfile('results', 'loss_surface.png');
outFig  = fullfile('results', 'loss_surface.fig');

% ---- 2) 读 CSV ----
T = readtable(csvPath);

% 必要列检查
needCols = {'dx','dy','loss_mean'};
for k = 1:numel(needCols)
    if ~ismember(needCols{k}, T.Properties.VariableNames)
        error("CSV 缺少列: %s", needCols{k});
    end
end

dx = T.dx;
dy = T.dy;
loss = T.loss_mean;

% ---- 3) 生成规则网格（sorted unique）----
xv = unique(dx, 'sorted');
yv = unique(dy, 'sorted');
[X, Y] = meshgrid(xv, yv);

% ---- 4) 把散点数据放进网格 Z ----
% 方法 A：用 accumarray（更稳，要求 (dx,dy) 是规则采样点）
[~, ix] = ismember(dx, xv);
[~, iy] = ismember(dy, yv);
Z = nan(numel(yv), numel(xv));
idx = sub2ind(size(Z), iy, ix);
Z(idx) = loss;

% 如果有缺失点（NaN），可选做插值填充
if any(isnan(Z(:)))
    fprintf("Warning: Z 中存在 NaN，尝试用 scatteredInterpolant 插值填充...\\n");
    F = scatteredInterpolant(dx, dy, loss, 'natural', 'none');
    Zi = F(X, Y);
    Z(isnan(Z)) = Zi(isnan(Z));
end

% ---- 5) 画 3D 曲面 ----
figure('Color','w');
surf(X, Y, Z, 'EdgeColor', 'none');  % 更平滑
colormap(parula);
colorbar;
xlabel('dx (perspective ratio)');
ylabel('dy (perspective ratio)');
zlabel('mean cross-entropy loss');
title('Loss Landscape under Perspective Perturbation');
view(-45, 35);
grid on;
camlight headlight; lighting gouraud;

% ---- 6) 也画一个等高线（可选）----
figure('Color','w');
contourf(X, Y, Z, 30, 'LineColor', 'none');
colormap(parula);
colorbar;
xlabel('dx (perspective ratio)');
ylabel('dy (perspective ratio)');
title('Loss Contour (mean loss)');
axis image;

% ---- 7) 保存 ----
% 图1：3D 曲面
figSurf = figure(1);
outPngSurf = fullfile('results', 'loss_surface.png');
outEpsSurf = fullfile('results', 'loss_surface.eps');
outFigSurf = fullfile('results', 'loss_surface.fig');
saveas(figSurf, outPngSurf);
saveas(figSurf, outFigSurf);
print(figSurf, outEpsSurf, '-depsc', '-painters');

% 图2：等高线
figContour = figure(2);
outPngContour = fullfile('results', 'loss_contour.png');
outEpsContour = fullfile('results', 'loss_contour.eps');
outFigContour = fullfile('results', 'loss_contour.fig');
saveas(figContour, outPngContour);
saveas(figContour, outFigContour);
print(figContour, outEpsContour, '-depsc', '-painters');

fprintf("Saved:\n  %s\n  %s\n  %s\n  %s\n", outPngSurf, outEpsSurf, outPngContour, outEpsContour);