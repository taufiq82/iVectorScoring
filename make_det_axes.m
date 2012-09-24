function make_det_axes()
%function make_det()
%
%  make_det_axes creates a plot for displaying detection performance
%  with the axes scaled and labeled so that a normal Gaussian
%  distribution will plot as a straight line.
%
%    The y axis represents the miss probability.
%    The x axis represents the false alarm probability.
%
%  Creates a new figure, switches hold on, embellishes and returns handle.

pROC_limits = [0.0005 0.5];

pticks = [0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.3 0.4];
ticklabels = ['0.1';'0.2';'0.5';' 1 ';' 2 ';' 5 ';'10 ';'20 ';'30 ';'40 '];

axis('square');

set (gca, 'xlim', probit(pROC_limits));
set (gca, 'xtick', probit(pticks));
set (gca, 'xticklabel', ticklabels);
set (gca, 'xgrid', 'on');
xlabel ('False Alarm probability (in %)');


set (gca, 'ylim', probit(pROC_limits));
set (gca, 'ytick', probit(pticks));
set (gca, 'yticklabel', ticklabels);
set (gca, 'ygrid', 'on')
ylabel ('Miss probability (in %)')

end

