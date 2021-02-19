import os
import torch
from nlgeval import compute_metrics
from tqdm import tqdm
from model import utils


def eval(net, dataloader, beam_size):
    net.eval()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    sample_count = 0
    out_file = open("output.txt", "w")
    lab_file_names = ["labels{}.txt".format(i) for i in range(5)]
    lab_files = [open(f, "w") for f in lab_file_names]

    for i, data in enumerate(tqdm(dataloader)):
        image, _, all_labels = data
        image_cuda = image.cuda()
        outputs = net.forward_single_inference(image.cuda(), beam_size)

        sample_count += image.shape[0]
        for j in range(image.shape[0]):
            h = outputs[j]
            ls = []
            for p, l in enumerate(all_labels[j]):
                ll = l[1:]
                ls.append(ll)
                l_text = utils.get_output_text(net.vectorizer, ll)
                lab_files[p].write(l_text + os.linesep)
            out_text = utils.get_output_text(net.vectorizer, h)
            out_file.write(out_text + os.linesep)

    out_file.close()
    for l in lab_files:
        l.close()
    metrics_dict = compute_metrics(references=lab_file_names, hypothesis='output.txt', no_overlap=False, no_skipthoughts=True, no_glove=True)
    return metrics_dict