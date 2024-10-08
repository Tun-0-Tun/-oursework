fname = 'D:\CBIA\Time-lapse series\Reg data for Anoshina\Seq B1 - B4 RA\Series015_RA.tif';

young = 1e4;
poiss = 0.4;
triHmax = 15;
meshtype = 'inner';
pntsN = 100;
pntsScaleCorr = 1;

cellMsuffix = '_body.tif';
cellm = readimTiff([fname(1:end-4) cellMsuffix]);

%%
elapsed_time = tic;

cellmInit = cellm;

cellm1 = slice_ex(cellm, 0);
cellm1 = dip_image(label(cellm1, Inf, 100), 'bin');
cellm2 = slice_ex(cellm, 1);
cellm2 = dip_image(label(cellm2, Inf, 100), 'bin');

showBinImgsInChannels(cellm1, cellm2)

cellB1 = GetBoundaryIm(cellm1);
cellB2 = GetBoundaryIm(cellm2);

P1 = GetContour(cellB1, pntsN*pntsScaleCorr);
P2 = GetContour(cellB2, pntsN*pntsScaleCorr);

match_cont_verbosity_level = 0;
[WP1, ~] = MatchContoursDT(P2, cellm1, cellm2, cellB1, cellB2, match_cont_verbosity_level);
WP2 = P2;

save([fname(1:end-4) '_WP1.mat'], 'WP1', '-v7');
save([fname(1:end-4) '_WP2.mat'], 'WP2', '-v7');

[ff, c] = GetFlowFieldNoDup(WP2, WP1);

[ffX, ffY] = algo_ff_built_FEM_LAME(ff, c, size(double(cellB2)), young, poiss, triHmax, meshtype);

dipshow(dip_image(ffX), 'lin')
dipshow(dip_image(ffY), 'lin')

save([fname(1:end-4) '_ffX.mat'], 'ffX', '-v7');
save([fname(1:end-4) '_ffY.mat'], 'ffY', '-v7');

% you can read mat files in python and compare it with your result
% https://stackoverflow.com/questions/874461/read-mat-files-in-python

disp('Elasped time for FF construction:');
toc(elapsed_time);
