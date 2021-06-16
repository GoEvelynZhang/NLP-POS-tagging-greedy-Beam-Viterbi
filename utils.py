from sklearn.metrics import f1_score
import numpy as np
import itertools as it


def calculate_f1_score(actual, predicted):
    print(f'Mean F1 Score: {f1_score(actual, predicted, average="weighted"):.5f}')

def token_accuracy(actual, predicted):
    match = [1 for true, pred in zip(actual, predicted) if true == pred]
    print(f"Per Token Accuracy: {sum(match)/len(actual):.5f}")


def sentence_accuracy(actual, predicted):
    match, total = 0, 0
    is_match = False
    for true, pred in zip(actual, predicted):
        if true == "O":
            total += 1
            if is_match:
                match += 1
            is_match = True
        elif true != pred:
            is_match = False
    if is_match:
        match += 1
    print(f"Per Sentence Accuracy:{match/total:.5f}")


def confusion_matrix(data_x, actual, predicted, tagset):
    error_example = []
    conf_matrix = np.zeros(shape=(len(tagset), len(tagset)))

    data_x = list(it.chain(*data_x))

    for token_x, true, pred in zip(data_x, actual, predicted):
        idx = np.where(true == tagset)[0]
        if true == pred:
            conf_matrix[idx, idx] += 1
        else:
            conf_matrix[idx, np.where(tagset == pred)[0]] += 1
            error_example.append([token_x, true, pred])

    norm_matrix = np.zeros(shape=(len(tagset), len(tagset)))

    for idx, row in enumerate(conf_matrix):
        norm_row = row/np.sum(row)
        norm_matrix[idx, :] = norm_row

    return norm_matrix, conf_matrix, np.array(error_example)


def display_confusion_matrix(data_x, actual, predicted, tagset):
    tagset = np.array(list(tagset))
    norm_conf_mat, conf_mat, error_example = confusion_matrix(data_x, actual, predicted, tagset)

    try:
        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        plt.figure(figsize=(12, 12))
        ax = plt.gca()
        im = ax.imshow(norm_conf_mat)

        plt.xticks(ticks=np.arange(len(tagset)), labels=tagset, rotation='vertical', fontsize=12)
        plt.yticks(ticks=np.arange(len(tagset)), labels=tagset, rotation='horizontal', fontsize=12)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        # Generate confusion matrix with only verb with different types of nouns.
        # verb vs. nns, nn, nnps, nnp.
        vb_idx = np.where(np.array(tagset) == 'VB')
        nn_idx = np.where(np.array(tagset) == 'NN')
        nnps_idx = np.where(np.array(tagset) == 'NNPS')
        nns_idx = np.where(np.array(tagset) == 'NNS')
        nnp_idx = np.where(np.array(tagset) == 'NNP')

        vb_nouns_idx = np.concatenate((vb_idx, nn_idx, nns_idx, nnp_idx, nnps_idx))
        conf_submat = conf_mat[vb_idx, vb_nouns_idx].T
        norm_conf_submat = conf_submat/np.sum(conf_submat)
        print(conf_submat[0])
        print(norm_conf_submat)

        # Print error analysis.
        # UH, FW, RBR, NNPS, SYM.
        uh_idx = np.where(np.array(tagset) == 'UH')
        fw_idx = np.where(np.array(tagset) == 'FW')
        rbr_idx = np.where(np.array(tagset) == 'RBR')
        nnps_idx = np.where(np.array(tagset) == 'NNPS')
        sym_idx = np.where(np.array(tagset) == 'SYM')

        uh_acc = norm_conf_mat[uh_idx[0], uh_idx[0]]
        fw_acc = norm_conf_mat[fw_idx[0], fw_idx[0]]
        rbr_acc = norm_conf_mat[rbr_idx[0], rbr_idx[0]]
        nnps_acc = norm_conf_mat[nnps_idx[0], nnps_idx[0]]
        sym_acc = norm_conf_mat[sym_idx[0], sym_idx[0]]

        uh_error_idx = np.where(error_example[:, 1] == 'UH')[0]
        fw_error_idx = np.where(error_example[:, 1] == 'FW')[0]
        rbr_error_idx = np.where(error_example[:, 1] == 'RBR')[0]
        nnps_error_idx = np.where(error_example[:, 1] == 'NNPS')[0]
        sym_error_idx = np.where(error_example[:, 1] == 'SYM')[0]

        print(error_example.shape)
        print(error_example[sym_error_idx])
    
    except:
        print('Unable to print confusion matrix. Matplotlib is not installed on your environment!')
