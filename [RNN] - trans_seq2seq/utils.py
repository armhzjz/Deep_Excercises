# Utils

# Functions to process the pairs (i.e. sentences) into their language word indexes.
# Training function is also defined in this file.

import torch
import random
import time
import math


SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10


def indexesFromSentence(lang, sentence):
    '''
        Take a sentence and return an array of its words indexes.
    '''
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device):
    '''
        Make use of the function "indexesFromSentences". Then appends
        the token "EOS_token" to the array and creats a torch tenso of
        type torch.long with such indexes.
    '''
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPairs(pair, input_lang, output_lang, device="cpu"):
    '''
        Takes a pair, input and output lang classes and constructs
        and returns input and target tensors.
    '''
    input_tensor = tensorFromSentence(input_lang, pair[0], device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device)
    return (input_tensor, target_tensor)


def train(encoder, encoder_optimizer, decoder, decoder_optimizer, criterion, input_tensor, target_tensor,
            max_lenth=10, device='cpu', attention=False, teaching_forcing_ratio=0.0):
    '''
        Train seq2seq to a pair. Important to notice that this function trains a seq2seq consisting
        of an encoder and a simple decoder (i.e. DecoderRNN class on the notebook).
    '''
    enc_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    if attention:
        encoder_outputs = torch.zeros(max_lenth, encoder.hidden_size, device=device)

    loss = 0

    for si in range(input_length):
        enc_output, enc_hidden = encoder(input_tensor[si], enc_hidden)
        if attention:
            encoder_outputs[si] = enc_output[0, 0]
    
    dec_input = torch.tensor([[SOS_token]], device=device)
    dec_hidden = enc_hidden

    use_teaching_forcing = True if random.random() < teaching_forcing_ratio else False

    if use_teaching_forcing:
        # Teacher forcing: feed the target as the next input
        for di in range(target_length):
            if attention:
                dec_output, dec_hidden, decoder_attention = decoder(dec_input, dec_hidden, encoder_outputs)
            else:
                dec_output, dec_hidden = decoder(dec_input, dec_hidden)

            loss += criterion(dec_output, target_tensor[di])
            dec_input = target_tensor[di] # Teacher forcing
    else:
        # Without teacher forcing: the decoder will use its own predictions
        # as its next input
        for di in range(target_length):
            if attention:
                dec_output, dec_hidden, decoder_attention = decoder(dec_input, dec_hidden, encoder_outputs)
            else:
                dec_output, dec_hidden = decoder(dec_input, dec_hidden)

            topv, topi = dec_output.topk(1)
            dec_input = topi.squeeze().detach() # detach from history as input

            loss += criterion(dec_output, target_tensor[di])
            if dec_input.item() == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# Helper functions to print time elapsed and estimated time remaining
# given the current time and progress percentage.
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m*60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (-%s)' % (asMinutes(s), asMinutes(rs))


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH, device="cpu", attention=False):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device) # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        if attention:
            decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder, decoder_attentions[:di + 1]
        else:
            for di in range(max_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder