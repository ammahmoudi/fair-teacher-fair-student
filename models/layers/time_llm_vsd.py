import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset # TensorDataset for simple dummy data
import logging
import os
from math import sqrt
from transformers import (
    LlamaConfig, LlamaModel, LlamaTokenizer,
    GPT2Config, GPT2Model, GPT2Tokenizer,
    BertConfig, BertModel, BertTokenizer,
    DistilBertConfig, DistilBertModel, DistilBertTokenizer,
)
import transformers

from models.layers.Embed import PatchEmbedding
from models.layers.StandardNorm import Normalize
from utils import logger

transformers.logging.set_verbosity_error()

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x): # Expected x: [B, N_vars, FeaturesDim1, FeaturesDim2]
        return self.dropout(self.linear(self.flatten(x)))

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)
        self.head_norm = nn.LayerNorm(d_keys) # Added head_norm

    def forward(self, target_embedding, source_key_embedding, source_value_embedding):
        B_eff, L_target, _ = target_embedding.shape
        S_source, _ = source_key_embedding.shape
        H = self.n_heads
        
        queries_einsum = self.query_projection(target_embedding).view(B_eff, L_target, H, -1)
        keys_einsum = self.key_projection(source_key_embedding).view(S_source, H, -1)
        values_einsum = self.value_projection(source_value_embedding).view(S_source, H, -1)
        
        scale = 1.0 / sqrt(queries_einsum.shape[-1])
        scores = torch.einsum("blhe,she->bhls", queries_einsum, keys_einsum)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogrammed_emb = torch.einsum("bhls,shd->blhd", A, values_einsum)
        reprogrammed_emb = self.head_norm(reprogrammed_emb) # Apply head_norm
        reprogrammed_emb = reprogrammed_emb.reshape(B_eff, L_target, -1)
        return self.out_projection(reprogrammed_emb)

class Model(nn.Module):
    def __init__(self, configs): # patch_len and stride should be in configs
        super(Model, self).__init__()
        logger.debug("Initializing Model with configs: %s", {k:v for k,v in configs.items() if k not in ['content', 'description']})
        
        self.task_name = configs["task_name"]
        self.prediction_length = configs["prediction_length"]
        self.sequence_length = configs["sequence_length"]
        
        # Ensure essential keys are present
        self.d_ff_equiv = configs["d_ff"] # d_ff from original, should ideally be related to d_llm
        self.top_k_lags = configs.get("top_k", 5)
        self.d_llm = configs["llm_dim"]
        self.patch_len = configs["patch_len"]
        self.stride = configs["stride"]
        self.d_model_patch = configs["d_model"] # For PatchEmbedding output
        self.n_heads_reprogram = configs["n_heads"] # For ReprogrammingLayer
        self.dropout_val = configs["dropout"]
        self.enc_in = configs["enc_in"]

        self.fine_tune_llm = configs.get("fine_tune_llm", False)
        self.llm_model_name = configs["llm_model"]
        self.llm_path_cfg = configs.get("llm_path", None)
        self.llm_layers_to_use = configs["llm_layers"]
        self.num_reprogram_tokens = configs.get("num_reprogram_tokens", 30) # Example default
        self.norm_affine = configs.get("norm_affine", False)
        self.max_prompt_len = configs.get("max_prompt_len", 128) # Example default

        # Flags to control LLM output for distillation, set by trainer or config
        self.output_llm_hidden_states = configs.get("output_llm_hidden_states", False)
        self.output_llm_attentions = configs.get("output_llm_attentions", False)

        logger.info(f"Model Init: LLM={self.llm_model_name}, LLM_Dim={self.d_llm}, FineTuneLLM={self.fine_tune_llm}")
        logger.info(f"Outputting LLM Internals: Hidden={self.output_llm_hidden_states}, Attn={self.output_llm_attentions}")

        _llm_path_default_map = {
            "GPT2": "openai-community/gpt2", "DistilBERT": "distilbert/distilbert-base-uncased",
            "BERT": "google-bert/bert-base-uncased", "LLAMA": "huggyllama/llama-7b" # Add others if needed
        }
        _llm_path = self.llm_path_cfg if self.llm_path_cfg else _llm_path_default_map.get(self.llm_model_name)
        if not _llm_path: raise ValueError(f"LLM path for {self.llm_model_name} not specified or found in defaults.")

        common_hf_config_args = {
            'output_hidden_states': self.output_llm_hidden_states,
            'output_attentions': self.output_llm_attentions
        }

        logging.debug("Initializing LLM based on config")
        if configs["llm_model"] == "LLAMA":
            logging.debug("LLM model is LLAMA")
            # self.llama_config = LlamaConfig.from_pretrained("huggyllama/llama-7b")
            self.llama_config = LlamaConfig.from_pretrained("huggyllama/llama-7b")
            self.llama_config.num_hidden_layers = configs["llm_layers"]
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                # self.llm_model = LlamaModel.from_pretrained(
                #     "huggyllama/llama-7b",
                #     trust_remote_code=True,
                #     local_files_only=False,
                #     config=self.llama_config,
                # )
                self.llm_model = LlamaModel.from_pretrained(
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
            except EnvironmentError:
                logging.info("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )

            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    "huggyllama/llama-7b", trust_remote_code=True, local_files_only=True
                )
            except EnvironmentError:
                logging.info(
                    "Local tokenizer files not found. Attempting to download them..."
                )
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    local_files_only=False,
                )
        elif configs["llm_model"] == "GPT2":
            logging.debug("LLM model is GPT2")
            self.gpt2_config = GPT2Config.from_pretrained("openai-community/gpt2")
            self.gpt2_config.num_hidden_layers = configs["llm_layers"]
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
            except EnvironmentError:
                logging.info("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=False,
                )
            except EnvironmentError:
                logging.info(
                    "Local tokenizer files not found. Attempting to download them..."
                )
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=False,
                )
        elif configs["llm_model"] == "BERT":
            logging.debug("LLM model is BERT")
            self.bert_config = BertConfig.from_pretrained(
                "google-bert/bert-base-uncased"
            )
            self.bert_config.num_hidden_layers = configs["llm_layers"]
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    "google-bert/bert-base-uncased",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            except EnvironmentError:
                logging.info("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    "google-bert/bert-base-uncased",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    "google-bert/bert-base-uncased",
                    trust_remote_code=True,
                    local_files_only=False,
                )
            except EnvironmentError:
                logging.info(
                    "Local tokenizer files not found. Attempting to download them..."
                )
                self.tokenizer = BertTokenizer.from_pretrained(
                    "google-bert/bert-base-uncased",
                    trust_remote_code=True,
                    local_files_only=False,
                )

        else:
            raise NotImplementedError(f"LLM model {self.llm_model_name} not implemented yet.")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else '[PAD]'
            logger.info(f"Set tokenizer.pad_token to {self.tokenizer.pad_token}")
            self.llm_model.resize_token_embeddings(len(self.tokenizer))

        for param in self.llm_model.parameters():
            param.requires_grad = self.fine_tune_llm
        logger.info(f"LLM ({self.llm_model_name}) fine_tune_llm set to {self.fine_tune_llm}")
        
        self.description_prompt_text = configs.get("content", "Time series forecasting task.")
        self.dropout_layer = nn.Dropout(self.dropout_val)
        self.patch_embedding_layer = PatchEmbedding(self.d_model_patch, self.patch_len, self.stride, self.dropout_val)
        
        # Learnable source embeddings for reprogramming
        self.learnable_source_keys_for_reprogram = nn.Parameter(torch.randn(self.num_reprogram_tokens, self.d_llm))
        self.learnable_source_values_for_reprogram = nn.Parameter(torch.randn(self.num_reprogram_tokens, self.d_llm))

        self.reprogramming_layer = ReprogrammingLayer(self.d_model_patch, self.n_heads_reprogram, d_llm=self.d_llm, attention_dropout=self.dropout_val)
        
        self.num_patches = int((self.sequence_length - self.patch_len) / self.stride + 2)
        # self.head_nf should use d_llm as features per patch come from reprogrammed LLM space
        self.head_nf = self.d_llm * self.num_patches 
        # Original used self.d_ff * self.num_patches. Ensure d_ff in config is consistent with d_llm expectations.
        if self.d_ff_equiv != self.d_llm:
             logger.warning(f"Config 'd_ff' ({self.d_ff_equiv}) is different from actual d_llm ({self.d_llm}) used for head_nf. Consider aligning them.")


        self.output_projection_head = FlattenHead(self.enc_in, self.head_nf, self.prediction_length, head_dropout=self.dropout_val)
        self.normalize_layer = Normalize(self.enc_in, affine=self.norm_affine)
        logger.info("Model initialization finished.")

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name not in ["long_term_forecast", "short_term_forecast"]:
            logger.error(f"Task {self.task_name} not supported in forward method.")
            return None, {} 

        # Normalization: x_enc shape [B, T, N]
        x_enc_normalized = self.normalize_layer(x_enc, "norm")
        B, T, N_vars = x_enc_normalized.shape

        # --- Prompt Creation (Copied from your Model's forecast/get_llm_representations) ---
        x_enc_for_stats = x_enc_normalized.permute(0, 2, 1).reshape(B * N_vars, T, 1)
        min_values = torch.min(x_enc_for_stats, dim=1)[0]
        max_values = torch.max(x_enc_for_stats, dim=1)[0]
        medians = torch.median(x_enc_for_stats, dim=1).values
        lags = self.calculate_lags(x_enc_for_stats)
        trends = x_enc_for_stats.diff(dim=1).sum(dim=1)
        
        prompts_text_list = []
        for i in range(B * N_vars): # For each variable in each batch item
            prompt_text = (f"<|start_prompt|>Desc: {self.description_prompt_text[:50]}. Task:F{self.prediction_length}/H{self.sequence_length}. "
                           f"Stats:min={min_values[i].item():.2f},max={max_values[i].item():.2f},med={medians[i].item():.2f}, "
                           f"trend={'U' if trends[i].item()>1e-3 else ('D'if trends[i].item()<-1e-3 else 'S')}, "
                           f"lags:{','.join(map(str,lags[i].tolist()))}.<|<end_prompt>|>" )
            prompts_text_list.append(prompt_text)
        
        tokenized_prompts = self.tokenizer(
            prompts_text_list, return_tensors="pt", padding=True, truncation=True, max_length=self.max_prompt_len
        ).input_ids.to(x_enc.device) # Shape: [B*N_vars, PromptSeqLen]
        
        prompt_embeddings = self.llm_model.get_input_embeddings()(tokenized_prompts) # Shape: [B*N_vars, PromptSeqLen, D_llm]

        # --- Patching and Reprogramming ---
        # Input to patch_embedding_layer: [B, N_vars, T_seq_len]
        patched_x, _ = self.patch_embedding_layer(x_enc_normalized.permute(0, 2, 1).contiguous()) # Shape: [B*N_vars, NumPatches, D_model_patch]
        reprogrammed_patches = self.reprogramming_layer(
            patched_x, self.learnable_source_keys_for_reprogram, self.learnable_source_values_for_reprogram
        ) # Shape: [B*N_vars, NumPatches, D_llm]

        llm_input_embeddings = torch.cat([prompt_embeddings, reprogrammed_patches], dim=1) # Shape: [B*N_vars, TotalSeqLen, D_llm]
        
        # --- LLM Forward Pass ---
        # output_hidden_states and output_attentions are controlled by self.llm_config set during init
        llm_outputs_obj = self.llm_model(inputs_embeds=llm_input_embeddings)
        
        llm_last_hidden_state = llm_outputs_obj.last_hidden_state # Shape: [B*N_vars, TotalSeqLen, D_llm]
        
        # --- Output Head ---
        # Extract features corresponding to reprogrammed patches
        patch_output_features = llm_last_hidden_state[:, tokenized_prompts.shape[1]:, :] # Shape: [B*N_vars, NumPatches, D_llm]
        
        # Reshape for FlattenHead: [B, N_vars, D_llm, NumPatches] (as per original intent for head_nf = D_llm * NumPatches)
        head_input = patch_output_features.reshape(B, N_vars, self.num_patches, self.d_llm)
        head_input_permuted = head_input.permute(0, 1, 3, 2).contiguous() # Shape: [B, N_vars, D_llm, NumPatches]
        
        predictions = self.output_projection_head(head_input_permuted) # Shape: [B, N_vars, PredictionLength]
        predictions = predictions.permute(0, 2, 1).contiguous() # Shape: [B, PredictionLength, N_vars]
        
        # Denormalization
        predictions = self.normalize_layer(predictions, "denorm")

        # --- Package representations for distillation ---
        llm_representations = {}
        if self.output_llm_hidden_states and hasattr(llm_outputs_obj, 'hidden_states') and llm_outputs_obj.hidden_states:
            llm_representations['hidden_states'] = llm_outputs_obj.hidden_states
        if self.output_llm_attentions and hasattr(llm_outputs_obj, 'attentions') and llm_outputs_obj.attentions:
            llm_representations['attentions'] = llm_outputs_obj.attentions
            
        return predictions, llm_representations

    def calculate_lags(self, x_enc_var_series): # x_enc_var_series: [B_eff, T, 1]
        x_for_fft = x_enc_var_series.squeeze(-1) 
        B_eff, T_steps = x_for_fft.shape
        if T_steps <= 1: 
            return torch.zeros(B_eff, self.top_k_lags, dtype=torch.long, device=x_for_fft.device)
        
        q_fft = torch.fft.rfft(x_for_fft, n=T_steps, dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(q_fft), n=T_steps, dim=-1)
        
        max_lag_to_check = min(T_steps - 1, T_steps // 2) 
        actual_k = min(self.top_k_lags, max_lag_to_check)

        if actual_k < 1:
             default_lags_val = 1 if T_steps > 1 else 0 # Default lag if no valid ones
             default_lags_tensor = torch.full((B_eff, self.top_k_lags), default_lags_val, dtype=torch.long, device=x_for_fft.device)
             if T_steps > 1 and actual_k == 0 : # if T_steps=2, max_lag_to_check=1, actual_k could be 0 if top_k_lags is 0.
                 possible_lags = torch.arange(1, T_steps, device=x_for_fft.device)
                 if len(possible_lags) > 0:
                     num_to_fill = min(len(possible_lags), self.top_k_lags)
                     default_lags_tensor[:, :num_to_fill] = possible_lags[:num_to_fill].unsqueeze(0)

             return default_lags_tensor

        _, top_indices = torch.topk(corr[:, 1 : max_lag_to_check + 1], k=actual_k, dim=-1)
        actual_lags = top_indices + 1
        
        if actual_k < self.top_k_lags:
            # Pad with the last valid lag or a default if no valid lags found
            padding_val = actual_lags[:, -1:].clone() if actual_lags.numel() > 0 else torch.ones_like(actual_lags[:,:1]) 
            padding = padding_val.repeat(1, self.top_k_lags - actual_k)
            actual_lags = torch.cat([actual_lags, padding], dim=1)
            
        return actual_lags
