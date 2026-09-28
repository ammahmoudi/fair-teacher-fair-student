import logging
from math import sqrt
import torch
import torch.nn as nn
from transformers import (
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer,
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    DistilBertConfig,
    DistilBertModel,
    DistilBertTokenizer,
    AutoModel, AutoTokenizer
)
import transformers
from models.layers.StandardNorm import Normalize
from models.layers.Embed import PatchEmbedding

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        logging.debug(
            "Initializing FlattenHead with n_vars=%d, nf=%d, target_window=%d, head_dropout=%f",
            n_vars,
            nf,
            target_window,
            head_dropout,
        )
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        logging.debug("Forward pass in FlattenHead with input shape: %s", x.shape)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        logging.debug("Output shape after FlattenHead forward pass: %s", x.shape)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        logging.debug("Initializing Model with configs: %s", configs)

        self.task_name = configs["task_name"]
        self.prediction_length = configs["prediction_length"]
        self.sequence_length = configs["sequence_length"]
        self.d_ff = configs["d_ff"]
        self.top_k = 5
        self.d_llm = configs["llm_dim"]
        self.patch_len = configs["patch_len"]
        self.stride = configs["stride"]

        logging.debug("Initializing LLM based on config")

        # -----------------------
        # Large Model: LLaMA 7B
        # ~6.7B parameters
        # Decoder-only transformer; powerful but not edge-friendly
        # https://huggingface.co/huggyllama/llama-7b
        # -----------------------
        if configs["llm_model"] == "LLAMA":
            logging.debug("LLM model is LLAMA")
            self.llama_config = LlamaConfig.from_pretrained("huggyllama/llama-7b")
            self.llama_config.num_hidden_layers = configs["llm_layers"]
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            self.llm_model = LlamaModel.from_pretrained(
                "huggyllama/llama-7b",
                trust_remote_code=True,
                local_files_only=False,
                config=self.llama_config,
            )
            self.tokenizer = LlamaTokenizer.from_pretrained(
                "huggyllama/llama-7b", trust_remote_code=True, local_files_only=False
            )

        # -----------------------
        # Medium Model: GPT2
        # ~117M parameters
        # Decoder-only, fast and simple
        # https://huggingface.co/openai-community/gpt2
        # -----------------------
        elif configs["llm_model"] == "GPT2":
            logging.debug("LLM model is GPT2")
            self.gpt2_config = GPT2Config.from_pretrained("openai-community/gpt2")
            self.gpt2_config.num_hidden_layers = configs["llm_layers"]
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_model = GPT2Model.from_pretrained(
                "openai-community/gpt2",
                trust_remote_code=True,
                local_files_only=False,
                config=self.gpt2_config,
            )
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                "openai-community/gpt2",
                trust_remote_code=True,
                local_files_only=False,
            )

        # -----------------------
        # Medium Model: BERT
        # ~110M parameters
        # Bidirectional encoder, well-known NLP baseline
        # https://huggingface.co/google-bert/bert-base-uncased
        # -----------------------
        elif configs["llm_model"] == "BERT":
            logging.debug("LLM model is BERT")
            self.bert_config = BertConfig.from_pretrained("google-bert/bert-base-uncased")
            self.bert_config.num_hidden_layers = configs["llm_layers"]
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            self.llm_model = BertModel.from_pretrained(
                "google-bert/bert-base-uncased",
                trust_remote_code=True,
                local_files_only=False,
                config=self.bert_config,
            )
            self.tokenizer = BertTokenizer.from_pretrained(
                "google-bert/bert-base-uncased",
                trust_remote_code=True,
                local_files_only=False,
            )

        # -----------------------
        # Small Model: DistilBERT
        # ~66M parameters
        # Light distilled BERT encoder, good for speed
        # https://huggingface.co/distilbert/distilbert-base-uncased
        # -----------------------
        elif configs["llm_model"] == "DistilBERT":
            logging.debug("LLM model is DistilBERT")
            self.distilbert_config = DistilBertConfig.from_pretrained(
                "distilbert/distilbert-base-uncased"
            )
            self.distilbert_config.n_layers = configs["llm_layers"]
            self.distilbert_config.output_attentions = True
            self.distilbert_config.output_hidden_states = True
            self.llm_model = DistilBertModel.from_pretrained(
                "distilbert/distilbert-base-uncased",
                trust_remote_code=True,
                local_files_only=False,
                config=self.distilbert_config,
            )
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert/distilbert-base-uncased",
                trust_remote_code=True,
                local_files_only=False,
            )

        # -----------------------
        # NEW: MiniLM
        # ~33M parameters
        # Distilled and very fast; high efficiency
        # https://huggingface.co/nreimers/MiniLMv2-L6-H384-distilled-from-BERT
        # -----------------------
        elif configs["llm_model"] == "MiniLM":
            self.llm_model = AutoModel.from_pretrained(
                "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large",
                trust_remote_code=True,
                local_files_only=False
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large",
                trust_remote_code=True,
                local_files_only=False
            )

        # -----------------------
        # NEW: TinyBERT
        # ~14M parameters
        # Super compact, distilled from BERT
        # https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D
        # -----------------------
        elif configs["llm_model"] == "TinyBERT":
            self.llm_model = AutoModel.from_pretrained(
                "huawei-noah/TinyBERT_General_4L_312D",
                trust_remote_code=True,
                local_files_only=False
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "huawei-noah/TinyBERT_General_4L_312D",
                trust_remote_code=True,
                local_files_only=False
            )

        # -----------------------
        # NEW: MobileBERT
        # ~25M parameters
        # Optimized for mobile inference
        # https://huggingface.co/google/mobilebert-uncased
        # -----------------------
        elif configs["llm_model"] == "MobileBERT":
            self.llm_model = AutoModel.from_pretrained(
                "google/mobilebert-uncased",
                trust_remote_code=True,
                local_files_only=False
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "google/mobilebert-uncased",
                trust_remote_code=True,
                local_files_only=False
            )

        # -----------------------
        # NEW: ALBERT
        # ~12M–18M parameters
        # Compact BERT-like model with shared layers
        # https://huggingface.co/albert-base-v2 or albert-tiny
        # -----------------------
        elif configs["llm_model"] == "ALBERT":
            self.llm_model = AutoModel.from_pretrained(
                "albert/albert-base-v2",
                trust_remote_code=True,
                local_files_only=False
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "albert/albert-base-v2",
                trust_remote_code=True,
                local_files_only=False
            )

        # -----------------------
        # NEW: BERT-tiny
        # ~4.4M parameters
        # Extremely small, fastest BERT variant
        # https://huggingface.co/prajjwal1/bert-tiny
        # -----------------------
        elif configs["llm_model"] == "BERT-tiny":
            self.llm_model = AutoModel.from_pretrained(
                "prajjwal1/bert-tiny",
                trust_remote_code=True,
                local_files_only=False
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "prajjwal1/bert-tiny",
                trust_remote_code=True,
                local_files_only=False
            )

        # -----------------------
        # NEW: OPT-125M
        # ~125M parameters
        # Lightweight GPT-style decoder from Meta
        # https://huggingface.co/facebook/opt-125m
        # -----------------------
        elif configs["llm_model"] == "OPT-125M":
            self.llm_model = AutoModel.from_pretrained(
                "facebook/opt-125m",
                trust_remote_code=True,
                local_files_only=False
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "facebook/opt-125m",
                trust_remote_code=True,
                local_files_only=False
            )
        

        else:
            raise Exception("LLM model is not defined")

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs["prompt_domain"]:
            self.description = configs["content"]
        else:
            self.description = "The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment."

        self.dropout = nn.Dropout(configs["dropout"])
        self.patch_embedding = PatchEmbedding(
            configs["d_model"], self.patch_len, self.stride, configs["dropout"]
        )
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(
            configs["d_model"], configs["n_heads"], self.d_ff, self.d_llm
        )
        self.patch_nums = int((configs["sequence_length"] - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            self.output_projection = FlattenHead(
                configs["enc_in"],
                self.head_nf,
                self.prediction_length,
                head_dropout=configs["dropout"],
            )
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs["enc_in"], affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        logging.debug("Model forward pass")
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.prediction_length :, :]
        return None, None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        logging.debug("Starting forecast function")
        x_enc = self.normalize_layers(x_enc, "norm")

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        logging.debug("Generated statistics for forecast prompts")
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.prediction_length)} steps given the previous {str(self.sequence_length)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(
            prompt.to(x_enc.device)
        )
        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0)
        ).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(
            enc_out, source_embeddings, source_embeddings
        )
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        llm_output = self.llm_model(inputs_embeds=llama_enc_out, output_attentions=True)
        dec_out = llm_output.last_hidden_state
        attn_weights = llm_output.attentions  # Extract attention maps

        dec_out = dec_out[:, :, : self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1])
        )
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums :])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers(dec_out, "denorm")

        logging.debug("Forecasting complete with output shape: %s", dec_out.shape)
        return dec_out

    def calcute_lags(self, x_enc):
        logging.debug("Calculating lags")
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        logging.debug("Lags calculated")
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(
        self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1
    ):
        super(ReprogrammingLayer, self).__init__()
        logging.debug("Initializing ReprogrammingLayer")
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        logging.debug("ReprogrammingLayer forward pass")
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        logging.debug("ReprogrammingLayer output shape: %s", out.shape)
        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1.0 / sqrt(E)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        logging.debug("Reprogramming complete")
        return reprogramming_embedding
