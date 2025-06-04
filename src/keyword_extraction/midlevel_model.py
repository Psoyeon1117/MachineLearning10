"""
Mid-level 모델: BiLSTM/GRU를 이용한 키워드 추출 (개선된 버전)
Hugging Face의 사전 훈련된 모델 사용, 통일된 불용어 시스템
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Tuple
import re
from .utils import preprocess_text # Assuming utils.py is in the same directory or src
from .english_stopwords import is_valid_english_keyword # Assuming english_stopwords.py is in the same directory or src

class MidLevelBaseExtractor:
    def __init__(self, model_name: str, tokenizer_name: str = None):
        self.device = torch.device('cpu')
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name else model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Handle cases where pad_token might be missing (e.g., for some GPT models)
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    print(f"Tokenizer for {self.tokenizer_name} missing pad_token. Using eos_token as pad_token.")
                else:
                    # Add a generic pad token if eos is also missing. This is less ideal.
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.model.resize_token_embeddings(len(self.tokenizer)) # Resize model embeddings
                    print(f"Tokenizer for {self.tokenizer_name} missing pad_token and eos_token. Added '[PAD]' as pad_token.")

            self.model.to(self.device)
            self.model.eval()
            print(f"Base model for MidLevelExtractor loaded successfully: {self.model_name} with tokenizer {self.tokenizer_name}")
        except Exception as e:
            print(f"Failed to load model {self.model_name} or tokenizer {self.tokenizer_name}: {e}")
            raise

    def _save_aggregated_word(self, word_str: str, embeddings_list: List[torch.Tensor], word_embeddings_dict: Dict[str, np.ndarray]):
        """
        Helper function to average embeddings for a word and save if valid.
        Uses min_length=2 as per original usage in simple extractors.
        """
        clean_word = word_str.strip().lower()
        if is_valid_english_keyword(clean_word, min_length=2) and embeddings_list:
            avg_embedding = torch.stack(embeddings_list).mean(dim=0)
            word_embeddings_dict[clean_word] = avg_embedding.cpu().numpy()

    def _extract_aggregated_word_embeddings(self, text: str) -> Dict[str, np.ndarray]:
        """
        Extracts word embeddings by aggregating subword token embeddings.
        Adapted from advanced_model.py's _extract_clean_word_embeddings.
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs_on_device)
                token_embeddings_tensor = outputs.last_hidden_state[0]  # [seq_len, hidden_size]

            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

            word_embeddings = {}
            current_word = ""
            current_embeddings_list = []

            for i, token_str in enumerate(tokens):
                if token_str in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token, self.tokenizer.unk_token]:
                    if current_word and current_embeddings_list: # End of a word
                        self._save_aggregated_word(current_word, current_embeddings_list, word_embeddings)
                    current_word = ""
                    current_embeddings_list = []
                    continue

                # Subword handling (BERT uses '##', RoBERTa/GPT use 'Ġ' or other indicators)
                # This heuristic tries to cover common cases.
                # For RoBERTa/XLM-R/GPT-2, 'Ġ' indicates a new word starting with a space.
                # For BERT, '##' indicates a subword part of the previous token.
                is_new_word = True
                token_to_append = token_str

                if 'roberta' in self.tokenizer.name_or_path.lower() or \
                   'gpt' in self.tokenizer.name_or_path.lower() or \
                   'xlm' in self.tokenizer.name_or_path.lower():
                    if token_str.startswith('Ġ'):
                        token_to_append = token_str[1:] # Remove 'Ġ'
                    elif i > 0 and tokens[i-1] not in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                        # If not starting with 'Ġ' and not the first token after special ones, assume continuation
                        is_new_word = False
                elif token_str.startswith('##'): # BERT-like
                    token_to_append = token_str[2:]
                    is_new_word = False
                
                # Other tokenizers might just split words and not use explicit subword markers visible here easily.
                # Fallback: if it's an unknown case and we have a current word, we treat it as a new word.

                if is_new_word:
                    if current_word and current_embeddings_list: # Save previous word
                        self._save_aggregated_word(current_word, current_embeddings_list, word_embeddings)
                    current_word = token_to_append
                    current_embeddings_list = [token_embeddings_tensor[i]]
                else: # Subword or continuation
                    current_word += token_to_append
                    current_embeddings_list.append(token_embeddings_tensor[i])
            
            # Save the last accumulated word
            if current_word and current_embeddings_list:
                self._save_aggregated_word(current_word, current_embeddings_list, word_embeddings)
            
            return word_embeddings
            
        except Exception as e:
            print(f"Word embedding aggregation failed for model {self.model_name}: {e}")
            return {}


class BiLSTMKeywordExtractor(MidLevelBaseExtractor):
    def __init__(self, model_name: str = "klue/roberta-base"):
        """
        BiLSTM-like Keyword Extractor using RoBERTa for bidirectional context.
        """
        try:
            super().__init__(model_name=model_name)
            print(f"BiLSTMKeywordExtractor (using {model_name}) initialized.")
        except Exception as e:
            print(f"KLUE RoBERTa ({model_name}) load failed, attempting fallback: {e}")
            try:
                super().__init__(model_name="bert-base-multilingual-cased")
                print(f"BiLSTMKeywordExtractor (using bert-base-multilingual-cased as fallback) initialized.")
            except Exception as e2:
                print(f"Fallback model (bert-base-multilingual-cased) also failed for BiLSTM: {e2}")
                # As a last resort, try a simpler model if available or raise
                try:
                    super().__init__(model_name="distilbert-base-multilingual-cased")
                    print(f"BiLSTMKeywordExtractor (using distilbert-base-multilingual-cased as final fallback) initialized.")
                except Exception as e3:
                    print(f"Final fallback model also failed for BiLSTM: {e3}")
                    raise # Or handle by setting self.model = None and failing gracefully in extract_keywords

    def calculate_importance_scores(self, word_embeddings: Dict[str, np.ndarray], text: str) -> Dict[str, float]:
        """Calculates word importance scores based on embeddings and text features."""
        if not word_embeddings:
            return {}
        
        word_frequencies = self._calculate_word_frequencies(text, list(word_embeddings.keys()))
        importance_scores = {}
        
        all_embedding_values = list(word_embeddings.values())
        if not all_embedding_values: return {} # Handle empty embeddings after filtering
        
        center_embedding = np.mean(all_embedding_values, axis=0) if all_embedding_values else np.zeros_like(next(iter(all_embedding_values)))


        for word, embedding in word_embeddings.items():
            freq_score = word_frequencies.get(word, 0)
            magnitude = np.linalg.norm(embedding)
            distance = np.linalg.norm(embedding - center_embedding)
            distinctiveness = self._calculate_word_distinctiveness(embedding, word_embeddings, word)
            
            length_bonus = 1.0
            if 3 <= len(word) <= 8: length_bonus = 1.2
            elif len(word) < 3: length_bonus = 0.6
            
            importance = (freq_score * 0.5 +
                         magnitude * 0.2 +
                         distance * 0.2 +
                         distinctiveness * 0.1) * length_bonus
            importance_scores[word] = importance
        
        return importance_scores

    def _calculate_word_frequencies(self, text: str, words: List[str]) -> Dict[str, float]:
        text_lower = text.lower()
        word_frequencies = {}
        for word in words:
            try: # Regex to match whole words
                count = len(re.findall(r'\b' + re.escape(word.lower()) + r'\b', text_lower))
            except re.error: # Handle potential regex errors with special characters in words
                count = text_lower.count(word.lower())

            if count > 0: word_frequencies[word] = 1 + np.log(count)
            else: word_frequencies[word] = 0.1
        return word_frequencies

    def _calculate_word_distinctiveness(self, target_embedding: np.ndarray, 
                                       all_embeddings: Dict[str, np.ndarray], 
                                       target_word: str) -> float:
        similarities = []
        if len(all_embeddings) <= 1 : return 0.5 # Avoid division by zero or meaningless distinctiveness

        for word, embedding in all_embeddings.items():
            if word != target_word:
                # Cosine similarity
                sim = np.dot(target_embedding, embedding) / (np.linalg.norm(target_embedding) * np.linalg.norm(embedding) + 1e-8)
                similarities.append(abs(sim))
        
        return (1.0 - np.mean(similarities)) if similarities else 0.5

    def extract_keywords(self, document: str, top_k: int = 5) -> List[str]:
        if not hasattr(self, 'model') or self.model is None:
            print("Model not loaded for BiLSTMKeywordExtractor, cannot extract keywords.")
            return []
            
        processed_doc = preprocess_text(document)
        if not processed_doc: return []
        
        try:
            word_embeddings = self._extract_aggregated_word_embeddings(processed_doc)
            if not word_embeddings: return []
            
            importance_scores = self.calculate_importance_scores(word_embeddings, processed_doc)
            if not importance_scores: return []
            
            sorted_words = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            
            keywords = []
            seen_words = set()
            for word, score in sorted_words:
                if word not in seen_words and len(keywords) < top_k : # is_valid_english_keyword already applied in aggregation
                    keywords.append(word)
                    seen_words.add(word)
            return keywords
        except Exception as e:
            print(f"BiLSTM keyword extraction failed: {e}")
            return []

class GRUKeywordExtractor(MidLevelBaseExtractor):
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", tokenizer_name: str = None):
        """
        GRU-like Keyword Extractor, attempting to use GPT-style models for sequential nature.
        """
        actual_tokenizer_name = tokenizer_name if tokenizer_name else model_name
        try:
            super().__init__(model_name=model_name, tokenizer_name=actual_tokenizer_name)
            print(f"GRUKeywordExtractor (using {model_name}) initialized.")
        except Exception as e:
            print(f"DialoGPT ({model_name}) load failed, attempting RoBERTa fallback: {e}")
            try:
                super().__init__(model_name="roberta-base", tokenizer_name="roberta-base") # RoBERTa is a robust fallback
                print(f"GRUKeywordExtractor (using roberta-base as fallback) initialized.")
            except Exception as e2:
                print(f"RoBERTa fallback failed for GRU, attempting DistilBERT: {e2}")
                try:
                    super().__init__(model_name="distilbert-base-uncased", tokenizer_name="distilbert-base-uncased")
                    print(f"GRUKeywordExtractor (using distilbert-base-uncased as final fallback) initialized.")
                except Exception as e3:
                    print(f"Final fallback model also failed for GRU: {e3}")
                    raise


    def _get_scores_from_embeddings(self, text: str) -> Dict[str, float]:
        """Calculates scores for words based on the norm of their aggregated embeddings."""
        word_embeddings = self._extract_aggregated_word_embeddings(text)
        word_scores = {}
        for word, embedding_np_array in word_embeddings.items():
            score = np.linalg.norm(embedding_np_array)
            word_scores[word] = score
        return word_scores

    def extract_keywords(self, document: str, top_k: int = 5) -> List[str]:
        if not hasattr(self, 'model') or self.model is None:
            print("Model not loaded for GRUKeywordExtractor, cannot extract keywords.")
            return []

        processed_doc = preprocess_text(document)
        if not processed_doc: return []
        
        try:
            word_scores = self._get_scores_from_embeddings(processed_doc)
            if not word_scores: return []
            
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            
            keywords = []
            seen_words = set()
            for word, score in sorted_words:
                 if word not in seen_words and len(keywords) < top_k: # is_valid_english_keyword already applied
                    keywords.append(word)
                    seen_words.add(word)
            return keywords
        except Exception as e:
            print(f"GRU keyword extraction failed: {e}")
            return []

def extract_keywords_midlevel(document: str, method: str = 'bilstm', top_k: int = 5) -> List[str]:
    """
    Mid-level 키워드 추출 함수 (개선됨)
    
    Args:
        document: 입력 문서
        method: 'bilstm' 또는 'gru'
        top_k: 추출할 키워드 수
    
    Returns:
        추출된 키워드 리스트
    """
    try:
        if method == 'bilstm':
            # Initialize with default model, or allow passing model_name if desired
            extractor = BiLSTMKeywordExtractor() 
        elif method == 'gru':
            extractor = GRUKeywordExtractor()
        else:
            print(f"Unknown midlevel method: {method}. Defaulting to bilstm.")
            extractor = BiLSTMKeywordExtractor()
        
        keywords = extractor.extract_keywords(document, top_k)
        return keywords
    except Exception as e:
        print(f"Mid-level keyword extraction failed for method {method}: {e}")
        return []

# 테스트 함수
if __name__ == "__main__":
    # Assuming utils.py is in the parent directory or accessible in PYTHONPATH
    # For direct execution, adjust path or ensure utils is discoverable
    try:
        from utils import load_data, get_random_samples, print_results # If run from parent dir of src
    except ImportError:
        from .utils import load_data, get_random_samples, print_results # If run as part of a package


    df = load_data() # Uses default path in utils.py
    if df is not None:
        samples = get_random_samples(df, 2)
        
        for doc_idx, document_content in samples:
            print(f"\n--- Testing Document Index: {doc_idx} ---")
            
            # BiLSTM 방법
            print("\nRunning BiLSTM Mid-level (Improved)...")
            keywords_bilstm = extract_keywords_midlevel(document_content, 'bilstm', 5)
            # print_results(doc_idx, document_content, keywords_bilstm, "BiLSTM Mid-level (Improved)")
            # Custom print for clarity during this specific test
            print(f"Document (first 100 chars): {document_content[:100]}...")
            print(f"Keywords (BiLSTM): {keywords_bilstm}")
            
            # GRU 방법
            print("\nRunning GRU Mid-level (Improved)...")
            keywords_gru = extract_keywords_midlevel(document_content, 'gru', 5)
            # print_results(doc_idx, document_content, keywords_gru, "GRU Mid-level (Improved)")
            print(f"Document (first 100 chars): {document_content[:100]}...")
            print(f"Keywords (GRU): {keywords_gru}")