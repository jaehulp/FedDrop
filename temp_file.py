def data_iter_plots(model, model_list, interaction_tensor, dataloader, num_classes): # Figure 3(c), disagreement rate

    model.eval()

    acc_list = []
    pred_mat = []
    correct_mat = []
    conf_mat = []

    for i, m in enumerate(model_list):
        print(f'data_iter model {i+1}th in')
        pred_list = []
        correct_list = []
        conf_list = []
        acc1_meter = AverageMeter()

        model.load_state_dict(m)

        for j, (xs, ys) in enumerate(dataloader):

            xs = xs.cuda()
            ys = ys.cuda()

            logits = model(xs)
            _, pred = logits.topk(1, 1, True, True)
            acc1 = accuracy(logits, ys)[0]
            acc1_meter.update(acc1)

            pred = pred.t()
            correct = pred.eq(ys.reshape(1, -1).expand_as(pred))
            pred_list.append(pred.squeeze().detach())
            correct_list.append(correct.squeeze().detach())

            softmax_output = torch.nn.functional.softmax(logits, dim=1)
            conf =  softmax_output[torch.arange(softmax_output.size(0)), ys]
            conf = torch.amax(softmax_output, dim=1)
            conf_list.append(conf.squeeze().detach())

        acc_list.append(acc1_meter.result())
        pred_list = torch.cat(pred_list)
        pred_mat.append(pred_list)
        correct_list = torch.cat(correct_list)
        correct_mat.append(correct_list)
        conf_list = torch.cat(conf_list)
        conf_mat.append(conf_list)
    correct_mat = torch.stack(correct_mat).bool().int()
    pred_mat = torch.stack(pred_mat)
    err_mat = ~(correct_mat.bool())
    conf_mat = torch.stack(conf_mat)

    pred_mat_1 = pred_mat.unsqueeze(0)
    pred_mat_2 = pred_mat.unsqueeze(1)
    pred_mat = (pred_mat_1 != pred_mat_2)

    dis_mat = torch.sum(pred_mat, dim=2) / pred_mat.shape[2]
    disagreement = dis_mat[torch.triu(torch.ones_like(dis_mat, dtype=bool), diagonal=1)]

    acc_list = torch.tensor(acc_list)
    print(acc_list)
    acc_mat = (acc_list.unsqueeze(1) + acc_list.unsqueeze(0)) / 2
    test_err = (100 - acc_mat) /100
    acc_list = test_err[torch.triu(torch.ones_like(test_err, dtype=bool), diagonal=1)] # Disagreement rate computation

    err_mat_1 = err_mat.unsqueeze(0)
    err_mat_2 = err_mat.unsqueeze(1)

    err_intersect = ((err_mat_1 == 1) & (err_mat_2 == 1)).int()
    err_intersect = torch.sum(err_intersect, dim=2)

    err_union = ((err_mat_1 == 1) | (err_mat_2 == 1)).int()
    err_union = torch.sum(err_union, dim=2)

    shared_err = err_intersect / err_union

    feature = torch.sum(interaction_tensor, dim=1).bool().int()
    feature_1 = feature.unsqueeze(0)
    feature_2 = feature.unsqueeze(1)

    feature_intersect = ((feature_1 == 1) & (feature_2 == 1)).int()
    feature_intersect = torch.sum(feature_intersect, dim=2)

    shared_err = shared_err[torch.triu(torch.ones_like(shared_err, dtype=bool), diagonal=1)]
    shared_feature = feature_intersect[torch.triu(torch.ones_like(feature_intersect, dtype=bool), diagonal=1)] # shared_err vs shared_feature comptutation

    conf_data = torch.mean(conf_mat, dim=0)
    num_features = torch.sum(torch.sum(interaction_tensor, dim=0), dim=1)

    M, N, T = interaction_tensor.shape

    labels = torch.tensor(dataloader.dataset.targets)
    interaction_tensor_class = torch.zeros((M, num_classes, int(N/num_classes), T))
    correct_mat_class= torch.zeros((M, num_classes, int(N/num_classes)))

    for i in range(num_classes):
        idx = (labels == i).nonzero(as_tuple=True)[0]
        interaction_tensor_class[:, i, :, :] = interaction_tensor[:, idx, :]
        correct_mat_class[:, i, :] = correct_mat[:, idx]

    feature_num_class = torch.sum(torch.sum(interaction_tensor_class, dim=0), dim=1).bool().int()
    class_feature_frequency = torch.sum(feature_num_class, dim=0)
    feature_num_class = torch.sum(feature_num_class, dim=1)
    class_feature_frequency, _ = torch.sort(class_feature_frequency, descending=True)

    num_feature_per_mc = torch.sum(interaction_tensor_class, dim=2).bool().int()
    num_feature_per_mc = torch.sum(num_feature_per_mc, dim=2)

    correct_per_mc = torch.sum(correct_mat_class, dim=2) / (N / num_classes)

    return disagreement.cpu(), acc_list.cpu(), shared_err.cpu(), shared_feature.cpu(), conf_data.cpu(), num_features.cpu(), feature_num_class.cpu(), class_feature_frequency.cpu(), num_feature_per_mc.cpu(), correct_per_mc.cpu()
