function handle = prepare_det(options)

if exist('options','var') && ~isempty(options)
    handle = figure(options{:});
else
    handle = figure();
end;
hold on;
make_det_axes();