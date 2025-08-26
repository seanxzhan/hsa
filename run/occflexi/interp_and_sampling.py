# ------------ shape interpolation ------------
if args.interp:
    # ------------ given bbox geom, complete shape ------------
    # NOTE: batch_node_feat has bbox sizes from model_id
    from utils import visualize
    white_bg = True
    it = args.it
    src_model_idx = args.src_idx
    tgt_model_idx = args.tgt_idx
    src_anno_id = model_idx_to_anno_id[src_model_idx]
    tgt_anno_id = model_idx_to_anno_id[tgt_model_idx]
    src_model_id = misc.anno_id_to_model_id(partnet_index_path)[src_anno_id]
    tgt_model_id = misc.anno_id_to_model_id(partnet_index_path)[tgt_anno_id]
    print(f"src anno id: {src_anno_id}, src model id: {src_model_id}")
    print(f"tgt anno id: {tgt_anno_id}, tgt model id: {tgt_model_id}")
    anon_str = '-'.join([src_anno_id, tgt_anno_id])
    results_dir = os.path.join(results_dir, 'interp', anon_str)
    misc.check_dir(results_dir)
    print("results dir: ", results_dir)

    # ------------ loading model, embedding, and data ------------
    if it is None or it == -1:
        checkpoint = torch.load(best_ckpt_path)
        print(f"best checkpoint occurred at iteration {checkpoint['epoch']}")
    else:
        checkpoint = torch.load(os.path.join(ckpt_dir, f'model_{it}.pt'))
    occ_model.load_state_dict(checkpoint['model_state_dict'])
    occ_embeddings = torch.nn.Embedding(num_shapes, num_parts*each_part_feat).to(device)
    occ_embeddings.load_state_dict(checkpoint['occ_embeddings_state_dict'])
    _, _, _, src_occ_embed, src_node_feat, src_adj, src_part_nodes, _, _ =\
        load_batch(0, 0, src_model_idx, src_model_idx+1)
    _, _, _, tgt_occ_embed, tgt_node_feat, tgt_adj, tgt_part_nodes, _, _ =\
        load_batch(0, 0, tgt_model_idx, tgt_model_idx+1)
    
    # # ------------ define bbox_geom ------------
    # anchor_bbox_geom = torch.einsum('ijk, ikm -> ijm',
    #                                 batch_part_nodes.to(torch.float32),
    #                                 batch_node_feat)[0].cpu().numpy()
    # ------------ making query points ------------
    query_points = reconstruct.make_query_points(pt_sample_res)
    query_points = torch.from_numpy(query_points).to(device, torch.float32)
    query_points = query_points.unsqueeze(0)
    bs, num_points, _ = query_points.shape

    num_interp = 10

    # ------------ reconstruct given sampled geometry ------------
    for interp_idx in range(num_interp+1):
        print(f"interpolating {interp_idx+1}/{num_interp+1}")
        interp_results_dir = os.path.join(results_dir, str(interp_idx))
        misc.check_dir(interp_results_dir)
        # batch_occ_embed = complete_occ_embed[sample_idx].unsqueeze(0)
        t = interp_idx*(1/num_interp)
        batch_occ_embed = (1-t)*src_occ_embed + t*tgt_occ_embed

        recon_one_shape(src_anno_id, interp_results_dir, args,
                        batch_occ_embed, src_adj, tgt_part_nodes,
                        eval=False, recon_gt=False)

    exit(0)

# ------------ sampling ------------
if args.samp:
    # ------------ sample bbox structure (geom+xform) and geometry ------------
    # NOTE: batch_node_feat has bbox sizes from model_id
    from utils import visualize
    white_bg = True
    it = args.it
    model_idx = args.test_idx
    anno_id = model_idx_to_anno_id[model_idx]
    model_id = misc.anno_id_to_model_id(partnet_index_path)[anno_id]
    print(f"anno id: {anno_id}, model id: {model_id}")
    top_level_results_dir = results_dir
    results_dir = os.path.join(results_dir, 'samp', anno_id)
    misc.check_dir(results_dir)
    print("results dir: ", results_dir)

    # ------------ loading model, embedding, and data ------------
    if it is None or it == -1:
        checkpoint = torch.load(best_ckpt_path)
        print(f"best checkpoint occurred at iteration {checkpoint['epoch']}")
    else:
        checkpoint = torch.load(os.path.join(ckpt_dir, f'model_{it}.pt'))
    occ_model.load_state_dict(checkpoint['model_state_dict'])
    occ_embeddings = torch.nn.Embedding(num_shapes, num_parts*each_part_feat).to(device)
    occ_embeddings.load_state_dict(checkpoint['occ_embeddings_state_dict'])
    _, _, _, batch_occ_embed, _, batch_adj, batch_part_nodes, _, _ =\
        load_batch(0, 0, model_idx, model_idx+1)
    
    # ------------ sample a geometry embedding ------------
    from sklearn.decomposition import PCA
    def sample_pca_space(xformed_pca, num_samples=10, scale=1.0):
        mean = np.mean(xformed_pca, axis=0)
        std_dev = np.std(xformed_pca, axis=0)
        samples = np.random.normal(mean, std_dev * scale, size=(num_samples, xformed_pca.shape[1]))
        return samples
    np.random.seed(319)
    num_samples = 30
    scale = 1.25  # Adjust scale to control the diversity of generated shapes
    pca = PCA(n_components=4)  # Number of components should be <= embedding_dim
    pca.fit(occ_embeddings.weight.data.cpu().numpy()[:num_shapes])
    occ_embedddings_pca = pca.transform(occ_embeddings.weight.data.cpu().numpy()[:num_shapes])
    samp_pca_occ_embeddings = sample_pca_space(occ_embedddings_pca, num_samples, scale)
    samp_lat_occ_embeddings = pca.inverse_transform(samp_pca_occ_embeddings)
    samp_lat_occ_embeddings = torch.from_numpy(samp_lat_occ_embeddings)
    
    _, _, _, _, name_to_obbs, _, _, _, _, _ =\
        preprocess_data_19.merge_partnet_after_merging(anno_id)

    # ------------ reconstruct given a sampled geometry embedding ------------
    # given occ, network optimizes for bbox transformation given occ
    for sample_idx in range(num_samples):
        samp_batch_occ_embed = samp_lat_occ_embeddings[sample_idx].to(device, torch.float32).unsqueeze(0)
        print(f"sampling {sample_idx+1}/{num_samples}")
        samp_results_dir = os.path.join(results_dir, str(sample_idx))
        misc.check_dir(samp_results_dir)
        recon_one_shape(anno_id, samp_results_dir, args,
                        samp_batch_occ_embed, batch_adj, batch_part_nodes,
                        eval=False)

    exit(0)