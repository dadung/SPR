clear;
addpath('utils');
addpath('hmm');
addpath('evaluation');
addpath('libs/flann-1.8.4-src/src/matlab');


work_dir = 'work_dir';
result_dir = [work_dir '/result'];

base_names{1} = '2014-06-26-09-24-58';
base_names{2} = '2014-06-26-08-53-56';

query_names{1} = '2014-06-23-15-41-25';
query_names{2} = '2014-06-23-15-36-04';
query_names{3} = '2014-06-23-15-14-44';
query_names{4} = '2014-06-24-14-15-17';


%% HMM parameters
hmm_params.W = 10;
hmm_params.distance_upper_bound = 2.0; 
hmm_params.large_distance = 2.5; 
hmm_params.sigma = 0.3; 

%% Update parameters
update_params.comb_node_prob = 0.3;
update_params.do_cull = false;
update_params.do_combine = false;

%% FLANN parameters
flann.algorithm = 'kmeans';
flann.branching = 16; 
flann.iterations = 50;
flann.centers_init = 'kmeanspp';
flann.checks = 128;
flann.knn = 5;
flann_build_parameters = struct('algorithm', flann.algorithm, ...
        'branching', flann.branching, 'iterations', flann.iterations, ...
        'centers_init', flann.centers_init);

%% Load database & build graph
tic;
[base_info, database, num_imgs_per_seq] = loadData(work_dir, base_names);
fprintf('Build graph...\n');
[node_list, aff_mat] = constructGraph(database, base_info, ...
    hmm_params.W, num_imgs_per_seq);
clearvars database base_info

fprintf('\tElapsed time = %.2fs\n', toc);

%% Do PR
for qur_idx = 1 : length(query_names)
    fprintf('\nQUERY = %s\n', query_names{qur_idx});
    
    %% Building indexing using FLANN
    if exist('database_index', 'var'), flann_free_index(database_index); end
    fprintf('Build indexing...\n');
    tic;
    [database, node_IDs] = constructFeatureMatrixFromGraph(node_list);

    database_index = flann_build_index(database, flann_build_parameters);
    fprintf('\t---Finished in %.2fs\n', toc);
    
    qur_name{1} = query_names{qur_idx};
    [query_info, query] = loadData(work_dir, qur_name);
    
    %% Find K-nearest neighbors
    fprintf('Find K-NN...\n');
    tic;
    query_length = size(query,2);
    
    [nearest_indices, nearest_distances] = flann_search(database_index, ...
        query, flann.knn, struct('checks',flann.checks));
    fprintf('\t---Finished in %.2fs\n', toc);
    
    %% Create observation model from K-NN
    fprintf('Make observation model from K-NN results...\n');
    tic;
    num_node = length(node_list);
    D = ones(query_length, num_node)*hmm_params.large_distance;
    D = exp(-D./hmm_params.sigma);
    
    uncertain_ids = find(nearest_distances > hmm_params.distance_upper_bound);
    nearest_distances(uncertain_ids) = hmm_params.large_distance;
    
    nearest_probs = exp(-nearest_distances./hmm_params.sigma);
    for ii = 1 : query_length
        nids = node_IDs(nearest_indices(:, ii)); % get node which store feature vectors
        D(ii, nids) = nearest_probs(:, ii);
    end
    fprintf('\t---Finished in %.2fs\n', toc);
    
    %% Perform Bayes Filter
    fprintf('Perform Bayes Filter...\n');
    tic;
    belief_all = doFilter(sparse(aff_mat), D'); 
    fprintf('\t---Finished in %.2fs\n', toc);
    
    %% Find and save matches
    fprintf('Find & save matches...\n');
    tic;
    pred = cell(query_length, 1);
    pred_file = [result_dir '/' qur_name{1} '.mat'];
    matches = -ones(query_length, 2);
    for T = 1 : query_length
        belief = belief_all(:,T);
        [matches(T,2), matches(T,1)]  = max(belief);
        matched_idx = matches(T,1);
        pred_loc = node_list{matched_idx}.loc(:,1);
        pred_rot = node_list{matched_idx}.rot(:,1);
        pred{T}.loc = pred_loc;
        pred{T}.rot = pred_rot;
    end
    save(pred_file, 'pred');
    fprintf('\t---Finished in %.2fs\n', toc);
    
    
    %% Do update & compression
    if update_params.do_cull
        fprintf('Cull new images to nodes...\n');
        tic;
        % Cull
        feas_id = size(database, 2) + 1;
        for ii = 1 : length(query_info)
            if ~isnan(matches(ii, 1))
                q_fea = query(:, ii);
                q_loc = query_info{ii}.loc;
                q_rot = query_info{ii}.rot;
                node_idx = matches(ii, 1);
                node_list{node_idx}.feas_id = [node_list{node_idx}.feas_id feas_id];
                node_list{node_idx}.feas = [node_list{node_idx}.feas q_fea];
                node_list{node_idx}.loc = [node_list{node_idx}.loc q_loc];
                node_list{node_idx}.rot = [node_list{node_idx}.rot q_rot];
            end
        end
        base_names = [base_names qur_name{1}];
        fprintf('\t---Finished in %.2fs\n', toc);
        
        % Combine nodes from belief
        if update_params.do_combine
            fprintf('Combine nodes from belief...\n');
            tic;
            fprintf('\tBefore combining: %d nodes\n', length(node_list));
            [node_list, aff_mat] = combineNodesFromBelief(node_list, aff_mat,belief_all, update_params.comb_node_prob);
            fprintf('\tAfter combining: %d nodes\n', length(node_list));
            fprintf('\t---Finished in %.2fs\n', toc);
        end

    
    end    
end
if exist('database_index', 'var'), flann_free_index(database_index); end
evaluate(query_names, work_dir, result_dir);

