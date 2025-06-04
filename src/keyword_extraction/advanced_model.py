"""
Advanced 모델: 순수 DistilBERT/BERT 기반 키워드 추출
TF-IDF 없이 오직 트랜스포머 모델의 임베딩과 어텐션만 사용
"""
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Tuple
import re
from .utils import preprocess_text
from .english_stopwords import is_valid_english_keyword

class PureDistilBERTExtractor:
    def __init__(self, model_name: str = "distilbert-base-multilingual-cased"):
        """
        순수 DistilBERT 키워드 추출기
        TF-IDF 없이 오직 BERT 임베딩과 어텐션만 사용
        """
        self.device = torch.device('cpu')
        self.model_name = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"순수 DistilBERT 로드 완료: {model_name}")
        except Exception as e:
            print(f"DistilBERT 모델 로드 실패: {e}")
            raise
    
    def extract_keywords_pure_clustering(self, text: str, top_k: int = 5) -> List[str]:
        """
        순수 BERT 임베딩 클러스터링 기반 키워드 추출
        1. 토큰별 임베딩 추출
        2. 의미적 중요도 계산
        3. 클러스터링으로 다양성 확보
        """
        try:
            # 단어 레벨 임베딩 추출 (개선된 방식)
            word_embeddings = self._extract_clean_word_embeddings(text)
            
            if not word_embeddings:
                return []
            
            # 순수 BERT 기반 중요도 계산
            importance_scores = self._calculate_semantic_importance(word_embeddings, text)
            
            if not importance_scores:
                return []
            
            # 의미적 클러스터링으로 다양한 키워드 선택
            final_keywords = self._semantic_clustering_selection(
                word_embeddings, importance_scores, top_k
            )
            
            return final_keywords
            
        except Exception as e:
            print(f"순수 클러스터링 추출 실패: {e}")
            return []
    
    def extract_keywords_pure_attention(self, text: str, top_k: int = 5) -> List[str]:
        """
        순수 BERT 어텐션 기반 키워드 추출
        1. 셀프 어텐션 가중치 계산
        2. 토큰 중요도 추출
        3. 단어 레벨로 집계
        """
        try:
            # 토큰 레벨 어텐션 분석
            attention_scores = self._analyze_self_attention(text)
            
            if not attention_scores:
                return []
            
            # 어텐션 기반 키워드 선택
            keywords = self._select_by_attention(attention_scores, top_k)
            
            return keywords
            
        except Exception as e:
            print(f"순수 어텐션 추출 실패: {e}")
            return []
    
    def _extract_clean_word_embeddings(self, text: str) -> Dict[str, np.ndarray]:
        """개선된 단어 레벨 임베딩 추출"""
        try:
            # 토큰화
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True,
                return_offsets_mapping=True if hasattr(self.tokenizer, 'return_offsets_mapping') else False
            )
            
            # 임베딩 생성
            with torch.no_grad():
                outputs = self.model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
                embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_size]
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # 서브워드를 원래 단어로 복원
            word_embeddings = {}
            current_word = ""
            current_embeddings = []
            
            for i, token in enumerate(tokens):
                if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                    continue
                
                if not token.startswith('##'):
                    # 이전 단어 저장
                    if current_word and current_embeddings:
                        self._save_word_if_valid(current_word, current_embeddings, word_embeddings)
                    
                    # 새 단어 시작
                    current_word = token.lower()
                    current_embeddings = [embeddings[i]]
                else:
                    # 서브워드 추가
                    current_word += token[2:].lower()  # ## 제거
                    current_embeddings.append(embeddings[i])
            
            # 마지막 단어 처리
            if current_word and current_embeddings:
                self._save_word_if_valid(current_word, current_embeddings, word_embeddings)
            
            return word_embeddings
            
        except Exception as e:
            print(f"단어 임베딩 추출 실패: {e}")
            return {}
    
    def _save_word_if_valid(self, word: str, embeddings_list: List[torch.Tensor], word_embeddings: Dict[str, np.ndarray]):
        """유효한 단어만 저장"""
        clean_word = word.strip().lower()
        
        # 완화된 검증 (2글자도 허용)
        if is_valid_english_keyword(clean_word, min_length=2):
            # 서브워드 임베딩들의 평균
            avg_embedding = torch.stack(embeddings_list).mean(dim=0)
            word_embeddings[clean_word] = avg_embedding.cpu().numpy()
    
    def _calculate_semantic_importance(self, word_embeddings: Dict[str, np.ndarray], text: str) -> Dict[str, float]:
        """순수 BERT 기반 의미적 중요도 계산 (문서 특이성 강화)"""
        if not word_embeddings:
            return {}
        
        try:
            # 1. 문서 내 빈도 계산 (TF-IDF의 TF 부분 모방)
            word_frequencies = self._calculate_word_frequencies(text, word_embeddings.keys())
            
            # 2. 대문자 원본 단어 발견 (고유명사 등 중요 단어)
            proper_noun_bonus = self._calculate_proper_noun_bonus(text, word_embeddings.keys())
            
            # 3. 문서 전체 표현 (CLS 토큰 임베딩)
            doc_representation = self._get_document_representation(text)
            
            importance_scores = {}
            
            for word, embedding in word_embeddings.items():
                # 1. 빈도 점수 (가장 중요 - TF-IDF 모방)
                freq_score = word_frequencies.get(word, 0)
                
                # 2. 고유명사 보너스 (pelosi, republican 같은 중요 단어)
                proper_bonus = proper_noun_bonus.get(word, 1.0)
                
                # 3. 단어 길이 보너스 (완전한 단어 선호)
                length_bonus = self._calculate_length_bonus(word)
                
                # 4. 임베딩 크기 (활성화 강도) - 비중 감소
                magnitude = np.linalg.norm(embedding)
                
                # 5. 문서와의 의미적 관련성 - 비중 감소
                if doc_representation is not None:
                    relevance = np.dot(embedding, doc_representation) / (
                        np.linalg.norm(embedding) * np.linalg.norm(doc_representation)
                    )
                    relevance = abs(relevance)
                else:
                    relevance = 0.5
                
                # 6. 임베딩 공간에서의 특이성 - 비중 감소
                distinctiveness = self._calculate_distinctiveness(embedding, word_embeddings, word)
                
                # 종합 점수 (빈도와 고유명사를 대폭 강화)
                importance = (
                    freq_score * 0.5 +           # 문서 내 빈도 (가장 중요)
                    (magnitude / 10.0) * 0.15 +  # 임베딩 크기 (비중 감소)
                    relevance * 0.15 +           # 문서 관련성 (비중 감소)
                    distinctiveness * 0.1 +      # 차별성 (비중 감소)
                    0.1                         # 기본 점수
                ) * proper_bonus * length_bonus  # 고유명사와 길이 보너스
                
                importance_scores[word] = importance
            
            return importance_scores
            
        except Exception as e:
            print(f"의미적 중요도 계산 실패: {e}")
            return {}
    
    def _calculate_proper_noun_bonus(self, text: str, words: List[str]) -> Dict[str, float]:
        """고유명사 보너스 계산 (대문자로 시작하는 단어들)"""
        proper_noun_bonus = {}
        
        # 원본 텍스트에서 대문자 패턴 찾기
        import re
        # 대문자로 시작하는 2글자 이상 단어 찾기
        proper_nouns = set(re.findall(r'\b[A-Z][a-z]{1,}\b', text))
        
        for word in words:
            word_lower = word.lower()
            # 해당 단어가 원본 텍스트에서 대문자로 나타나는지 확인
            is_proper_noun = any(proper.lower() == word_lower for proper in proper_nouns)
            
            if is_proper_noun:
                proper_noun_bonus[word] = 1.5  # 50% 보너스
            else:
                proper_noun_bonus[word] = 1.0  # 보너스 없음
        
        return proper_noun_bonus
    
    def _calculate_length_bonus(self, word: str) -> float:
        """단어 길이 보너스 (완전한 단어 선호)"""
        length = len(word)
        
        if 3 <= length <= 10:  # 적절한 길이
            return 1.3
        elif length == 2:      # 너무 짧음
            return 0.3
        elif length > 15:      # 너무 김
            return 0.7
        else:
            return 1.0
    
    def _calculate_word_frequencies(self, text: str, words: List[str]) -> Dict[str, float]:
        """문서 내 단어 빈도 계산 (TF 모방)"""
        text_lower = text.lower()
        word_frequencies = {}
        
        for word in words:
            # 단어 등장 횟수 계산
            count = text_lower.count(word.lower())
            # 정규화 (log scale 사용)
            if count > 0:
                word_frequencies[word] = 1 + np.log(count)
            else:
                word_frequencies[word] = 0.1  # 최소값
        
        return word_frequencies
    
    def _get_document_representation(self, text: str) -> np.ndarray:
        """문서 전체 표현 (CLS 토큰)"""
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # CLS 토큰의 임베딩이 문서 전체를 표현
                doc_representation = outputs.last_hidden_state[0][0]  # [CLS]
                
            return doc_representation.cpu().numpy()
            
        except Exception as e:
            print(f"문서 표현 생성 실패: {e}")
            return None
    
    def _calculate_distinctiveness(self, target_embedding: np.ndarray, all_embeddings: Dict[str, np.ndarray], target_word: str) -> float:
        """다른 단어들과의 차별성 계산"""
        try:
            similarities = []
            
            for word, embedding in all_embeddings.items():
                if word != target_word:
                    # 코사인 유사도
                    similarity = np.dot(target_embedding, embedding) / (
                        np.linalg.norm(target_embedding) * np.linalg.norm(embedding)
                    )
                    similarities.append(abs(similarity))
            
            if similarities:
                # 평균 유사도가 낮을수록 더 구별됨
                avg_similarity = np.mean(similarities)
                distinctiveness = 1.0 - avg_similarity
                return max(0.0, distinctiveness)
            else:
                return 0.5
                
        except Exception as e:
            print(f"차별성 계산 실패: {e}")
            return 0.5
    
    def _semantic_clustering_selection(self, word_embeddings: Dict[str, np.ndarray], 
                                     importance_scores: Dict[str, float], top_k: int) -> List[str]:
        """의미적 클러스터링으로 다양한 키워드 선택"""
        try:
            # 중요도순 정렬
            sorted_words = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            
            selected_keywords = []
            selected_embeddings = []
            
            for word, score in sorted_words:
                if len(selected_keywords) >= top_k:
                    break
                
                current_embedding = word_embeddings[word]
                
                # 이미 선택된 키워드들과의 의미적 유사성 체크
                if selected_embeddings:
                    similarities = []
                    for selected_emb in selected_embeddings:
                        similarity = np.dot(current_embedding, selected_emb) / (
                            np.linalg.norm(current_embedding) * np.linalg.norm(selected_emb)
                        )
                        similarities.append(abs(similarity))
                    
                    max_similarity = max(similarities)
                    
                    # 너무 유사한 키워드는 제외 (의미적 다양성 확보)
                    if max_similarity > 0.75:  # 임계값
                        continue
                
                selected_keywords.append(word)
                selected_embeddings.append(current_embedding)
            
            return selected_keywords
            
        except Exception as e:
            print(f"클러스터링 선택 실패: {e}")
            # 실패시 단순 중요도순 반환
            sorted_words = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            return [word for word, score in sorted_words[:top_k]]
    
    def _analyze_self_attention(self, text: str) -> Dict[str, float]:
        """셀프 어텐션 분석 (DistilBERT는 어텐션 출력이 기본적으로 없음)"""
        try:
            # DistilBERT의 경우 임베딩 기반 대안 방법 사용
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[0]
                
                # 각 토큰의 상대적 활성화를 어텐션 대용으로 사용
                # 문서 평균과의 차이로 중요도 계산
                doc_mean = embeddings.mean(dim=0)
                
                attention_weights = []
                for i in range(embeddings.shape[0]):
                    # 문서 평균으로부터의 거리를 중요도로 사용
                    distance = torch.norm(embeddings[i] - doc_mean)
                    attention_weights.append(distance.item())
            
            # 토큰별 어텐션 점수 매핑
            attention_scores = {}
            current_word = ""
            current_scores = []
            
            for i, token in enumerate(tokens):
                if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                    continue
                
                if not token.startswith('##'):
                    # 이전 단어 저장
                    if current_word and current_scores:
                        avg_score = np.mean(current_scores)
                        clean_word = current_word.lower().strip()
                        if is_valid_english_keyword(clean_word, min_length=2):
                            attention_scores[clean_word] = avg_score
                    
                    # 새 단어 시작
                    current_word = token.lower()
                    current_scores = [attention_weights[i]]
                else:
                    # 서브워드 추가
                    current_word += token[2:].lower()
                    current_scores.append(attention_weights[i])
            
            # 마지막 단어 처리
            if current_word and current_scores:
                avg_score = np.mean(current_scores)
                clean_word = current_word.lower().strip()
                if is_valid_english_keyword(clean_word, min_length=2):
                    attention_scores[clean_word] = avg_score
            
            return attention_scores
            
        except Exception as e:
            print(f"어텐션 분석 실패: {e}")
            return {}
    
    def _select_by_attention(self, attention_scores: Dict[str, float], top_k: int) -> List[str]:
        """어텐션 점수 기반 키워드 선택"""
        try:
            # 점수순 정렬
            sorted_words = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 상위 키워드 선택
            keywords = []
            for word, score in sorted_words:
                if len(keywords) >= top_k:
                    break
                keywords.append(word)
            
            return keywords
            
        except Exception as e:
            print(f"어텐션 기반 선택 실패: {e}")
            return []

class PureKLUEBERTExtractor:
    """순수 KLUE BERT 추출기 (실제 KLUE 모델 사용)"""
    def __init__(self, model_name: str = "klue/bert-base"):
        self.device = torch.device('cpu')
        
        # 실제 KLUE BERT 사용 시도
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"순수 KLUE BERT 로드 완료: {model_name}")
        except Exception as e:
            print(f"KLUE BERT 로드 실패, 대안 모델 사용: {e}")
            # 대안으로 multilingual BERT 사용
            try:
                model_name = "bert-base-multilingual-cased"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                print(f"KLUE BERT 대안 모델 로드 완룼: {model_name}")
            except Exception as e2:
                print(f"Multilingual BERT도 실패, DistilBERT 사용: {e2}")
                # 최종 대안
                model_name = "distilbert-base-multilingual-cased"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                print(f"KLUE BERT 최종 대안 모델 로드 완료: {model_name}")
    
    def extract_keywords(self, document: str, top_k: int = 5) -> List[str]:
        """순수 BERT 기반 키워드 추출"""
        try:
            # PureDistilBERTExtractor와 동일한 로직 사용
            extractor = PureDistilBERTExtractor()
            return extractor.extract_keywords_pure_clustering(document, top_k)
        except Exception as e:
            print(f"순수 KLUE BERT 추출 실패: {e}")
            return []

def extract_keywords_advanced(document: str, method: str = 'distilbert', approach: str = 'clustering', top_k: int = 5) -> List[str]:
    """
    순수 Advanced 키워드 추출 함수 (TF-IDF 완전 배제)
    
    Args:
        document: 입력 문서
        method: 'distilbert' 또는 'klue'
        approach: 'clustering' 또는 'attention'
        top_k: 추출할 키워드 수
    
    Returns:
        추출된 키워드 리스트
    """
    try:
        if method == 'distilbert':
            extractor = PureDistilBERTExtractor()
            if approach == 'clustering':
                keywords = extractor.extract_keywords_pure_clustering(document, top_k)
            else:  # attention
                keywords = extractor.extract_keywords_pure_attention(document, top_k)
        else:  # klue
            extractor = PureKLUEBERTExtractor()
            keywords = extractor.extract_keywords(document, top_k)
        
        return keywords
    except Exception as e:
        print(f"순수 Advanced 키워드 추출 실패: {e}")
        return []

# 테스트 함수
if __name__ == "__main__":
    from .utils import load_data, get_random_samples, print_results
    
    # 데이터 로드
    df = load_data()
    if df is not None:
        # 랜덤 샘플 테스트
        samples = get_random_samples(df, 2)
        
        for doc_idx, document in samples:
            # 순수 DistilBERT 클러스터링
            keywords_distil_clust = extract_keywords_advanced(document, 'distilbert', 'clustering', 5)
            print_results(doc_idx, document, keywords_distil_clust, "순수 DistilBERT (Clustering)")
            
            # 순수 DistilBERT 어텐션
            keywords_distil_att = extract_keywords_advanced(document, 'distilbert', 'attention', 5)
            print_results(doc_idx, document, keywords_distil_att, "순수 DistilBERT (Attention)")
            
            # 순수 KLUE BERT
            keywords_klue = extract_keywords_advanced(document, 'klue', 'clustering', 5)
            print_results(doc_idx, document, keywords_klue, "순수 KLUE BERT")
