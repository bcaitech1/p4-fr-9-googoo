import os
import argparse
import multiprocessing
import numpy as np
import random
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import yaml
from tqdm import tqdm
from checkpoint import (
    default_checkpoint,
    load_checkpoint,
    save_checkpoint,
    init_tensorboard,
    write_tensorboard,
)
from psutil import virtual_memory

from flags import Flags
from utils import get_network, get_optimizer
from dataset import dataset_loader, START, PAD,load_vocab
from scheduler import CircularLRBeta

from metrics import word_error_rate,sentence_acc
import wandb
import pdb

os.environ['WANDB_PROJECT'] = 'Pstage4_OCR'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def id_to_string(tokens, data_loader,do_eval=0):
    result = []
    if do_eval:
        special_ids = [data_loader.dataset.token_to_id["<PAD>"], data_loader.dataset.token_to_id["<SOS>"],
                       data_loader.dataset.token_to_id["<EOS>"]]

    for example in tokens:
        string = ""
        if do_eval:
            for token in example:
                token = token.item()
                if token not in special_ids:
                    if token != -1:
                        string += data_loader.dataset.id_to_token[token] + " "
        else:
            for token in example:
                token = token.item()
                if token != -1:
                    string += data_loader.dataset.id_to_token[token] + " "

        result.append(string)
    return result

def run_epoch(
    data_loader,
    model,
    epoch_text,
    criterion,
    optimizer,
    lr_scheduler,
    teacher_forcing_ratio,
    max_grad_norm,
    device,
    train=True,
):
    # Disables autograd during validation mode
    torch.set_grad_enabled(train)
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    grad_norms = []
    correct_symbols = 0
    total_symbols = 0
    wer=0
    num_wer=0
    sent_acc=0
    num_sent_acc=0

    with tqdm(
        desc="{} ({})".format(epoch_text, "Train" if train else "Validation"),
        total=len(data_loader.dataset),
        dynamic_ncols=True,
        leave=False,
    ) as pbar:
        for d in data_loader:
            '''(Pdb) !d
            {'path': ['/opt/ml/input/data/train_dataset/images/train_00019.jpg', 
            '/opt/ml/input/data/train_dataset/images/train_00012.jpg', '/opt/ml/input/data/train_dataset/images/train_00017.jpg', 
            '/opt/ml/input/data/train_dataset/images/train_00023.jpg', '/opt/ml/input/data/train_dataset/images/train_00005.jpg', 
            '/opt/ml/input/data/train_dataset/images/train_00021.jpg', '/opt/ml/input/data/train_dataset/images/train_00022.jpg', 
            '/opt/ml/input/data/train_dataset/images/train_00011.jpg', '/opt/ml/input/data/train_dataset/images/train_00001.jpg', 
            '/opt/ml/input/data/train_dataset/images/train_00020.jpg', '/opt/ml/input/data/train_dataset/images/train_00030.jpg', 
            '/opt/ml/input/data/train_dataset/images/train_00029.jpg', '/opt/ml/input/data/train_dataset/images/train_00000.jpg', 
            '/opt/ml/input/data/train_dataset/images/train_00031.jpg', '/opt/ml/input/data/train_dataset/images/train_00026.jpg', 
            '/opt/ml/input/data/train_dataset/images/train_00010.jpg', '/opt/ml/input/data/train_dataset/images/train_00009.jpg', 
            '/opt/ml/input/data/train_dataset/images/train_00007.jpg', '/opt/ml/input/data/train_dataset/images/train_00025.jpg', 
            '/opt/ml/input/data/train_dataset/images/train_00003.jpg', '/opt/ml/input/data/train_dataset/images/train_00028.jpg', 
            '/opt/ml/input/data/train_dataset/images/train_00002.jpg', '/opt/ml/input/data/train_dataset/images/train_00013.jpg', 
            '/opt/ml/input/data/train_dataset/images/train_00027.jpg', '/opt/ml/input/data/train_dataset/images/train_00015.jpg', 
            '/opt/ml/input/data/train_dataset/images/train_00018.jpg'], 
            'image': tensor([[[[0.6824, 0.6902, 0.6863,  ..., 0.6980, 0.6980, 0.6980],
            [0.6745, 0.6863, 0.6824,  ..., 0.6941, 0.6980, 0.7020],
            [0.6824, 0.6784, 0.6863,  ..., 0.6980, 0.6980, 0.6941],
            ...,
            [0.6980, 0.7020, 0.6980,  ..., 0.6902, 0.6902, 0.6824],
            [0.7020, 0.6980, 0.7020,  ..., 0.6902, 0.6902, 0.6902],
            [0.6941, 0.6941, 0.6980,  ..., 0.6863, 0.6902, 0.6863]]],


            [[[0.5882, 0.5529, 0.5686,  ..., 0.6941, 0.6980, 0.7020],
            [0.6000, 0.5608, 0.5412,  ..., 0.7020, 0.6941, 0.6941],
            [0.6039, 0.5373, 0.5216,  ..., 0.7098, 0.7020, 0.6863],
            ...,
            [0.2902, 0.3294, 0.3255,  ..., 0.6039, 0.6314, 0.5647],
            [0.3020, 0.3373, 0.3608,  ..., 0.6157, 0.6431, 0.5647],
            [0.3176, 0.3490, 0.3804,  ..., 0.5804, 0.5765, 0.5725]]],


            [[[0.5098, 0.5059, 0.4980,  ..., 0.4471, 0.4471, 0.4471],
            [0.5059, 0.5020, 0.5020,  ..., 0.4510, 0.4471, 0.4471],
            [0.5059, 0.5020, 0.4980,  ..., 0.4510, 0.4510, 0.4471],
            ...,
            [0.5020, 0.5020, 0.5137,  ..., 0.4314, 0.4314, 0.4275],
            [0.5020, 0.5020, 0.5059,  ..., 0.4353, 0.4353, 0.4314],
            [0.4941, 0.4941, 0.5020,  ..., 0.4392, 0.4314, 0.4314]]],


            ...,


            [[[0.4000, 0.4235, 0.4314,  ..., 0.4784, 0.4745, 0.4784],
            [0.3882, 0.4157, 0.4314,  ..., 0.4745, 0.4745, 0.4784],
            [0.3686, 0.4196, 0.4275,  ..., 0.4745, 0.4706, 0.4706],
            ...,
            [0.3059, 0.3882, 0.4392,  ..., 0.4706, 0.4627, 0.4706],
            [0.2745, 0.3843, 0.4392,  ..., 0.4745, 0.4706, 0.4706],
            [0.2549, 0.3922, 0.4392,  ..., 0.4745, 0.4706, 0.4745]]],


            [[[0.6588, 0.6588, 0.6667,  ..., 0.6667, 0.6667, 0.6667],
            [0.6627, 0.6667, 0.6706,  ..., 0.6667, 0.6627, 0.6627],
            [0.6667, 0.6667, 0.6745,  ..., 0.6667, 0.6667, 0.6667],
            ...,
            [0.6980, 0.6941, 0.6941,  ..., 0.6784, 0.6784, 0.6824],
            [0.6941, 0.6980, 0.7020,  ..., 0.6824, 0.6824, 0.6784],
            [0.6941, 0.6980, 0.7020,  ..., 0.6863, 0.6863, 0.6824]]],


            [[[0.7098, 0.7020, 0.7020,  ..., 0.5020, 0.5255, 0.5098],
            [0.7137, 0.7020, 0.6980,  ..., 0.5059, 0.5255, 0.5098],
            [0.7137, 0.7059, 0.6980,  ..., 0.5020, 0.5255, 0.5020],
            ...,
            [0.6980, 0.6980, 0.7098,  ..., 0.5255, 0.5216, 0.5216],
            [0.7020, 0.7059, 0.7098,  ..., 0.5176, 0.5176, 0.5176],
            [0.7059, 0.7020, 0.7020,  ..., 0.5176, 0.5137, 0.5216]]]]), 
            'truth': {'text': ['a _ { n } = a _ { 1 } r ^ { n - 1 }', '7 \\div 4', '= h \\left( a \\right)', 't = - 1', '\\sum \\overrightarrow { F } _ { e x t } = d', 
            '= 3 x \\left( x ^ { 2 } + 1 \\right)', '1 8 4 - \\left( 8 9 + 8 8 \\right) = 7', '2 2 + 7 - 1 2 =', 'a ^ { x } > q', 'x = - 2', 'P \\left( A \\right) \\cdot P
            \\left( B \\right) = \\frac { 1 } { 2 } \\times \\frac { 1 } { 2 } = \\frac { 1 } { 4 } \\neq \\left( A \\cap B \\right)', 'a = \\sqrt { a _ { r } ^ { 2 } + a _ 
            { \\theta } ^ { 2 } } = \\sqrt { \\left( 0 . 3 3 6 \\right) ^ { 2 } + \\left( 0 . 0 7 \\right) ^ { 2 } } = 0 . 3 4 m / s', '4 \\times 7 = 2 8', 'x = \\frac 
            { - b ^ \\prime \\pm \\sqrt { b ^ \\prime - a c } } { a }', 'x < 4', '\\therefore b = - 9', '\\left( a - 2 \\right) \\left( a - 3 \\right) = 0', '7 \\times 9 = 
            4 9', '\\frac { 1 6 } { 2 7 } \\div \\frac { 2 } { 2 7 } =', '\\sum _ { k = 1 } ^ { n - 1 } b _ { k } = a _ { n } - a _ { 1 }', '3 6 . 4 8 \\div 4 , 5 6', 
            '8 \\times 9', 'f \\left( x \\right) = 4 x ^ { 3 }', '= x \\left( x - 1 \\right) \\cdot \\left( x - 2 \\right) !', '\\sum F _ { \\theta } = m a _ { \\theta }', 
            '1 0 4 \\times 9 8 = \\left( 1 0 0 + 4 \\right) \\left( 1 0 0 - 2 \\right)'], 
            'encoded': tensor([[  0, 205,  10,  ...,  -1,  -1,  -1],
            [  0, 160, 121,  ...,  -1,  -1,  -1],
            [  0, 180, 176,  ...,  -1,  -1,  -1],
            ...,
            [  0, 180, 204,  ...,  -1,  -1,  -1],
            [  0, 117, 178,  ...,  -1,  -1,  -1],
            [  0, 125,  14,  ...,  -1,  -1,  -1]])}}
            '''
            
            input = d["image"].to(device)

            # The last batch may not be a full batch
            curr_batch_size = len(input)
            expected = d["truth"]["encoded"].to(device)

            # Replace -1 with the PAD token
            expected[expected == -1] = data_loader.dataset.token_to_id[PAD]
            '''(Pdb) expected  shape (batch_size, seq_len). seq_len은 해당 배치에서 가장 긴 target의 길이임.
            tensor([[  0, 205,  10,  ...,   2,   2,   2],
            [  0, 160, 121,  ...,   2,   2,   2],
            [  0, 180, 176,  ...,   2,   2,   2],
            ...,
            [  0, 180, 204,  ...,   2,   2,   2],
            [  0, 117, 178,  ...,   2,   2,   2],
            [  0, 125,  14,  ...,   2,   2,   2]], device='cuda:0')
            '''
            with torch.set_grad_enabled(train):
                output = model(input, expected, train, teacher_forcing_ratio) ## shape (batch_size, seq_len - 1, # vocab). <SOS>토큰은 예측하지 않으므로 seq_len-1이다.
                
                decoded_values = output.transpose(1, 2) ## shape (batch_size, # vocab, seq_len - 1)
                _, sequence = torch.topk(decoded_values, 1, dim=1) ## shape (batch_size, 1, seq_len - 1)
                sequence = sequence.squeeze(1) ## shape (batch_size, seq_len - 1)
                
                loss = criterion(decoded_values, expected[:, 1:])
                '''(Pdb) loss
                tensor(5.5526, device='cuda:0', grad_fn=<NllLoss2DBackward>)
                nn.crossentropyloss는 예측값을 (N, C, ...)와 같이 넣을 수 도 있다(C는 클래스 갯수). 그러므로 (batch_size, num_classes, max_len) 이렇게 넣어도 됨.
                이 때 target은 (batch_size, max_len)으로  넣어야 됨.
                (Pdb) criterion(output.reshape(-1, 245), expected[:,1:].reshape(-1))
                tensor(5.5526, device='cuda:0', grad_fn=<NllLossBackward>)
                '''

                if train:
                    optim_params = [
                        p
                        for param_group in optimizer.param_groups
                        for p in param_group["params"]
                    ]
                    optimizer.zero_grad()
                    loss.backward()
                    # Clip gradients, it returns the total norm of all parameters
                    grad_norm = nn.utils.clip_grad_norm_(
                        optim_params, max_norm=max_grad_norm
                    )
                    grad_norms.append(grad_norm) 

                    # cycle
                    lr_scheduler.step()
                    optimizer.step()

            losses.append(loss.item())
            
            expected[expected == data_loader.dataset.token_to_id[PAD]] = -1
            expected_str = id_to_string(expected, data_loader,do_eval=1)
            '''(Pdb) expected_str
            ['a _ { n } = a _ { 1 } r ^ { n - 1 } ', '7 \\div 4 ', '= h \\left( a \\right) ', 't = - 1 ', '\\sum \\overrightarrow { F } _ { e x t } = d ', 
            '= 3 x \\left( x ^ { 2 } + 1 \\right) ', '1 8 4 - \\left( 8 9 + 8 8 \\right) = 7 ', '2 2 + 7 - 1 2 = ', 'a ^ { x } > q ', 'x = - 2 ', 
            'P \\left( A \\right) \\cdot P \\left( B \\right) = \\frac { 1 } { 2 } \\times \\frac { 1 } { 2 } = \\frac { 1 } { 4 } \\neq \\left( A \\cap B \\right) ', 
            'a = \\sqrt { a _ { r } ^ { 2 } + a _ { \\theta } ^ { 2 } } = \\sqrt { \\left( 0 . 3 3 6 \\right) ^ { 2 } + \\left( 0 . 0 7 \\right) ^ { 2 } } = 0 . 3 4 m / s ',
             ...]
            '''
            sequence_str = id_to_string(sequence, data_loader,do_eval=1)
            '''['\\right\\| \\right\\| \\to \\right\\| \\to \\widehat $ \\to \\to \\widehat y 8 \\to \\widehat y 8 \\to \\widehat y 8 \\to \\widehat y 8 \\to \\widehat y 8 \\to \\widehat y 8 \\to \\widehat y 8 \\to \\widehat y 8 \\to \\widehat y 8 \\to \\widehat y 8 \\to \\widehat y 8 \\to \\widehat y 8 \\to \\widehat y ', 'j j \\cot 8 j j j j \\ddot j j j \\ddot j j j \\ddot j j j \\ddot j j j \\ddot j j j \\ddot j j j \\ddot j j j \\ddot j j j \\ddot j j j \\ddot j j j \\ddot j j j \\ddot j j j \\ddot j j ', 
            '\\Phi \\varphi \\right\\| \\right\\| \\right\\| \\perp V & \\perp V & \\vdots \\perp V & \\vdots \\perp V & \\vdots \\perp V & \\vdots \\perp V & \\vdots \\perp V & \\vdots \\perp V & \\vdots \\perp V & \\vdots \\perp V & \\vdots \\perp V & \\vdots \\perp V & \\vdots \\perp V & \\vdots \\perp V & ',
            ...]
            '''
            wer += word_error_rate(sequence_str,expected_str)
            num_wer += 1
            sent_acc += sentence_acc(sequence_str,expected_str)
            num_sent_acc += 1
            correct_symbols += torch.sum(sequence == expected[:, 1:], dim=(0, 1)).item()
            total_symbols += torch.sum(expected[:, 1:] != -1, dim=(0, 1)).item()

            pbar.update(curr_batch_size)

  

    expected = id_to_string(expected, data_loader)
    sequence = id_to_string(sequence, data_loader)
    print("-" * 10 + "GT ({})".format("train" if train else "valid"))
    print(*expected[:3], sep="\n")
    print("-" * 10 + "PR ({})".format("train" if train else "valid"))
    print(*sequence[:3], sep="\n")

    result = {
        "loss": np.mean(losses),
        "correct_symbols": correct_symbols,
        "total_symbols": total_symbols,
        "wer": wer,
        "num_wer":num_wer,
        "sent_acc": sent_acc,
        "num_sent_acc":num_sent_acc
    }
    if train:
        try:
            result["grad_norm"] = np.mean([tensor.cpu() for tensor in grad_norms])
        except:
            result["grad_norm"] = np.mean(grad_norms)
    return result


def main(config_file, options):
    """
    Train math formula recognition model
    """
    
    '''(Pdb) options
    FLAGS(Attention=FLAGS(cell_type='LSTM', embedding_dim=128, hidden_dim=128, layer_num=1, src_dim=512), SATRN=FLAGS(decoder=FLAGS(filter_dim=512, head_num=8, hidden_dim=128, 
    layer_num=3, src_dim=300), encoder=FLAGS(filter_dim=600, head_num=8, hidden_dim=300, layer_num=6)), batch_size=96, checkpoint='', 
    data=FLAGS(crop=True, dataset_proportions=[1.0], random_split=True, rgb=1, test=[''], test_proportions=0.2, token_paths=['/opt/ml/input/data/train_dataset/tokens.txt'],
    train=['/opt/ml/input/data/train_dataset/gt.txt']), dropout_rate=0.1, input_size=FLAGS(height=128, width=128), max_grad_norm=2.0, network='Attention', num_epochs=50,
    num_workers=8, optimizer=FLAGS(is_cycle=True, lr=0.0005, optimizer='Adam', weight_decay=0.0001), prefix='././log/attention_50', print_epochs=1, seed=1234, 
    teacher_forcing_ratio=0.5)
    '''

    #set random seed
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))

    # Print system environments
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024 ** 3)
    print(
        "[+] System environments\n",
        "The number of gpus : {}\n".format(num_gpus),
        "The number of cpus : {}\n".format(num_cpus),
        "Memory Size : {}G\n".format(mem_size),
    )

    # Load checkpoint and print result
    checkpoint = (
        load_checkpoint(options.checkpoint, cuda=is_cuda)
        if options.checkpoint != ""
        else default_checkpoint
    )
    model_checkpoint = checkpoint["model"]
    if model_checkpoint:
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
            "Train Symbol Accuracy : {:.5f}\n".format(checkpoint["train_symbol_accuracy"][-1]),
            "Train Sentence Accuracy : {:.5f}\n".format(checkpoint["train_sentence_accuracy"][-1]),
            "Train WER : {:.5f}\n".format(checkpoint["train_wer"][-1]),
            "Train Loss : {:.5f}\n".format(checkpoint["train_losses"][-1]),
            "Validation Symbol Accuracy : {:.5f}\n".format(
                checkpoint["validation_symbol_accuracy"][-1]
            ),
            "Validation Sentence Accuracy : {:.5f}\n".format(
                checkpoint["validation_sentence_accuracy"][-1]
            ),
            "Validation WER : {:.5f}\n".format(
                checkpoint["validation_wer"][-1]
            ),
            "Validation Loss : {:.5f}\n".format(checkpoint["validation_losses"][-1]),
        )

    # Get data
    transformed = transforms.Compose(
        [
            # Resize so all images have the same size
            transforms.Resize((options.input_size.height, options.input_size.width)),
            transforms.ToTensor(),
        ]
    )
    train_data_loader, validation_data_loader, train_dataset, valid_dataset = dataset_loader(options, transformed)
    print(
        "[+] Data\n",
        "The number of train samples : {}\n".format(len(train_dataset)),
        "The number of validation samples : {}\n".format(len(valid_dataset)),
        "The number of classes : {}\n".format(len(train_dataset.token_to_id)),
    )


    # Get loss, model
    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        train_dataset,
    )
    model.train()
    criterion = model.criterion.to(device)
    enc_params_to_optimise = [
        param for param in model.encoder.parameters() if param.requires_grad
    ]
    dec_params_to_optimise = [
        param for param in model.decoder.parameters() if param.requires_grad
    ]
    params_to_optimise = [*enc_params_to_optimise, *dec_params_to_optimise]
    print(
        "[+] Network\n",
        "Type: {}\n".format(options.network),
        "Encoder parameters: {}\n".format(
            sum(p.numel() for p in enc_params_to_optimise),
        ),
        "Decoder parameters: {} \n".format(
            sum(p.numel() for p in dec_params_to_optimise),
        ),
    )

    # Get optimizer
    optimizer = get_optimizer(
        options.optimizer.optimizer,
        params_to_optimise,
        lr=options.optimizer.lr,
        weight_decay=options.optimizer.weight_decay,
    )
    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = options.optimizer.lr
    if options.optimizer.is_cycle:
        cycle = len(train_data_loader) * options.num_epochs
        lr_scheduler = CircularLRBeta(
            optimizer, options.optimizer.lr, 10, 10, cycle, [0.95, 0.85]
        )
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=options.optimizer.lr_epochs,
            gamma=options.optimizer.lr_factor,
        )

    # Log
    if not os.path.exists(options.prefix):
        os.makedirs(options.prefix)
    # log_file = open(os.path.join(options.prefix, "log.txt"), "w")
    shutil.copy(config_file, os.path.join(options.prefix, "train_config.yaml"))
    if options.print_epochs is None:
        options.print_epochs = options.num_epochs
    # writer = init_tensorboard(name=options.prefix.strip("-"))
    start_epoch = checkpoint["epoch"]
    train_symbol_accuracy = checkpoint["train_symbol_accuracy"]
    train_sentence_accuracy=checkpoint["train_sentence_accuracy"]
    train_wer=checkpoint["train_wer"]
    train_losses = checkpoint["train_losses"]
    validation_symbol_accuracy = checkpoint["validation_symbol_accuracy"]
    validation_sentence_accuracy=checkpoint["validation_sentence_accuracy"]
    validation_wer=checkpoint["validation_wer"]
    validation_losses = checkpoint["validation_losses"]
    learning_rates = checkpoint["lr"]
    grad_norms = checkpoint["grad_norm"]

    # Train
    for epoch in range(options.num_epochs):
        start_time = time.time()


        epoch_text = "[{current:>{pad}}/{end}] Epoch {epoch}".format(
            current=epoch + 1,
            end=options.num_epochs,
            epoch=start_epoch + epoch + 1,
            pad=len(str(options.num_epochs)),
        )

        # Train
        train_result = run_epoch(
            train_data_loader,
            model,
            epoch_text,
            criterion,
            optimizer,
            lr_scheduler,
            options.teacher_forcing_ratio,
            options.max_grad_norm,
            device,
            train=True,
        )



        train_losses.append(train_result["loss"])
        grad_norms.append(train_result["grad_norm"])
        train_epoch_symbol_accuracy = (
            train_result["correct_symbols"] / train_result["total_symbols"]
        )
        train_symbol_accuracy.append(train_epoch_symbol_accuracy)
        train_epoch_sentence_accuracy = (
                train_result["sent_acc"] / train_result["num_sent_acc"]
        )

        train_sentence_accuracy.append(train_epoch_sentence_accuracy)
        train_epoch_wer = (
                train_result["wer"] / train_result["num_wer"]
        )
        train_wer.append(train_epoch_wer)
        epoch_lr = lr_scheduler.get_lr()  # cycle

        # Validation
        validation_result = run_epoch(
            validation_data_loader,
            model,
            epoch_text,
            criterion,
            optimizer,
            lr_scheduler,
            options.teacher_forcing_ratio,
            options.max_grad_norm,
            device,
            train=False,
        )
        validation_losses.append(validation_result["loss"])
        validation_epoch_symbol_accuracy = (
            validation_result["correct_symbols"] / validation_result["total_symbols"]
        )
        validation_symbol_accuracy.append(validation_epoch_symbol_accuracy)

        validation_epoch_sentence_accuracy = (
            validation_result["sent_acc"] / validation_result["num_sent_acc"]
        )
        validation_sentence_accuracy.append(validation_epoch_sentence_accuracy)
        validation_epoch_wer = (
                validation_result["wer"] / validation_result["num_wer"]
        )
        validation_wer.append(validation_epoch_wer)

        # Save checkpoint
        #make config
        with open(config_file, 'r') as f:
            option_dict = yaml.safe_load(f)

        save_checkpoint(
            {
                "epoch": start_epoch + epoch + 1,
                "train_losses": train_losses,
                "train_symbol_accuracy": train_symbol_accuracy,
                "train_sentence_accuracy": train_sentence_accuracy,
                "train_wer":train_wer,
                "validation_losses": validation_losses,
                "validation_symbol_accuracy": validation_symbol_accuracy,
                "validation_sentence_accuracy":validation_sentence_accuracy,
                "validation_wer":validation_wer,
                "lr": learning_rates,
                "grad_norm": grad_norms,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "configs": option_dict,
                "token_to_id":train_data_loader.dataset.token_to_id,
                "id_to_token":train_data_loader.dataset.id_to_token
            },
            prefix=options.prefix + options.version,
        )

        # Summary
        elapsed_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        if epoch % options.print_epochs == 0 or epoch == options.num_epochs - 1:
            output_string = (
                "{epoch_text}: "
                "Train Symbol Accuracy = {train_symbol_accuracy:.5f}, "
                "Train Sentence Accuracy = {train_sentence_accuracy:.5f}, "
                "Train WER = {train_wer:.5f}, "
                "Train Loss = {train_loss:.5f}, "
                "Validation Symbol Accuracy = {validation_symbol_accuracy:.5f}, "
                "Validation Sentence Accuracy = {validation_sentence_accuracy:.5f}, "
                "Validation WER = {validation_wer:.5f}, "
                "Validation Loss = {validation_loss:.5f}, "
                "lr = {lr} "
                "(time elapsed {time})"
            ).format(
                epoch_text=epoch_text,
                train_symbol_accuracy=train_epoch_symbol_accuracy,
                train_sentence_accuracy=train_epoch_sentence_accuracy,
                train_wer=train_epoch_wer,
                train_loss=train_result["loss"],
                validation_symbol_accuracy=validation_epoch_symbol_accuracy,
                validation_sentence_accuracy=validation_epoch_sentence_accuracy,
                validation_wer=validation_epoch_wer,
                validation_loss=validation_result["loss"],
                lr=epoch_lr,
                time=elapsed_time,
            )
            print(output_string)
            wandb.log(
                {
                    'epoch_text':epoch_text,
                    'train_symbol_accuracy':train_epoch_symbol_accuracy,
                    'train_sentence_accuracy':train_epoch_sentence_accuracy,
                    'train_wer':train_epoch_wer,
                    'train_loss':train_result["loss"],
                    'validation_symbol_accuracy':validation_epoch_symbol_accuracy,
                    'validation_sentence_accuracy':validation_epoch_sentence_accuracy,
                    'validation_wer':validation_epoch_wer,
                    'validation_loss':validation_result["loss"],
                    'lr':epoch_lr,
                    'time':elapsed_time,

                })
            # log_file.write(output_string + "\n")
            # write_tensorboard(
            #     writer,
            #     start_epoch + epoch + 1,
            #     train_result["grad_norm"],
            #     train_result["loss"],
            #     train_epoch_symbol_accuracy,
            #     train_epoch_sentence_accuracy,
            #     train_epoch_wer,
            #     validation_result["loss"],
            #     validation_epoch_symbol_accuracy,
            #     validation_epoch_sentence_accuracy,
            #     validation_epoch_wer,
            #     model,
            # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        default="configs/SATRN.yaml",
        type=str,
        help="Path of configuration file",
    )
    parser.add_argument(
        "--version",
        default="_v1",
        type=str,
        help="version",
    )
    parser = parser.parse_args()
    options = Flags(parser.config_file).get()
    network = parser.config_file.split('/')[1].split('.')[0]
    wandb.init(tags=['ignore padding index', 'batch_size 34', 'epoch 30 to 60'], name = network + options.version)
    main(parser.config_file, options)
