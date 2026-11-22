function matlab_olps_replay(olps, dest, names)
% Run the unmodified, pinned up_run.m and save each completed repetition.
addpath(fullfile(olps, 'Strategy'));
assert(strcmp(which('up_run'), fullfile(olps, 'Strategy', 'up_run.m')));
opts = struct('quiet_mode', true, 'progress', false);
if ispc, sink = 'NUL'; else, sink = '/dev/null'; end
fid = fopen(sink, 'w');
cleanup = onCleanup(@() fclose(fid));
runtime = version;
for k = 1:numel(names)
    name = names{k};
    market = load(fullfile(dest, 'cache', [name '_native_input.mat']));
    x = market.x;
    for rep = 1:3
        file = fullfile(dest, 'cache', sprintf('%s_matlab_%d.mat', name, rep));
        if isfile(file), continue; end
        rng(724191, 'twister');
        tic;
        [terminal, cumulative, gains, w] = up_run(fid, x, 0, opts);
        seconds = toc;
        assert(all(isfinite(w(:))) && min(w(:)) >= 0);
        assert(max(abs(sum(w, 2) - 1)) < 1e-12 && all(gains > 0));
        assert(max(abs(sum(w .* x, 2) - gains)) < 1e-12);
        assert(max(abs(cumsum(log(gains)) - log(cumulative))) < 1e-10);
        temp = [file '.part.mat'];
        save(temp, 'w', 'terminal', 'cumulative', 'seconds', 'gains', ...
             'runtime', 'rep', '-v7');
        movefile(temp, file);
        fprintf('%s repetition %d complete, %.3f seconds\n', name, rep, seconds);
    end
end
end
