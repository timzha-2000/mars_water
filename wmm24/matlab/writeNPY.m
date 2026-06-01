function writeNPY(arr, filename)
% writeNPY  Minimal NumPy .npy writer (float64) for 1-D / 2-D arrays.
% Writes in Fortran order so MATLAB's column-major memory is used as-is;
% numpy reads fortran_order=True transparently.
arr = double(arr);
shp = size(arr);
if isvector(arr)
    shp = numel(arr);              % store 1-D vectors as shape (N,)
    fortran = 'False';
else
    fortran = 'True';
end

% Build shape tuple string, e.g. "(123,)" or "(123, 6)"
if numel(shp) == 1
    shape_str = sprintf('(%d,)', shp);
else
    parts = arrayfun(@(v) sprintf('%d', v), shp, 'UniformOutput', false);
    shape_str = ['(' strjoin(parts, ', ') ')'];
end

header = sprintf("{'descr': '<f8', 'fortran_order': %s, 'shape': %s, }", ...
                 fortran, shape_str);
header = char(header);

% Pad so that (10 + len(header)+1) is a multiple of 64; header ends in \n
total = 10 + length(header) + 1;
pad = mod(64 - mod(total, 64), 64);
header = [header repmat(' ', 1, pad) newline];

fid = fopen(filename, 'wb');
if fid < 0, error('Cannot open %s', filename); end
fwrite(fid, uint8([147 78 85 77 80 89]), 'uint8');   % \x93 N U M P Y
fwrite(fid, uint8([1 0]), 'uint8');                  % version 1.0
fwrite(fid, uint16(length(header)), 'uint16');       % header length (LE)
fwrite(fid, uint8(header), 'uint8');                 % header bytes (ASCII)
fwrite(fid, arr, 'double');                          % column-major == Fortran order
fclose(fid);
end
