class Gramformer:

  def __init__(self, models=1, use_gpu=False):
    from transformers import AutoTokenizer
    from transformers import AutoModelForSeq2SeqLM
    #from lm_scorer.models.auto import AutoLMScorer as LMScorer
    import errant
    self.annotator = errant.load('en')
    
    if use_gpu:
        device= "cuda:0"
    else:
        device = "cpu"
    batch_size = 1    
    #self.scorer = LMScorer.from_pretrained("gpt2", device=device, batch_size=batch_size)    
    self.device    = device
    correction_model_tag = "prithivida/grammar_error_correcter_v1"
    self.model_loaded = False

    if models == 1:
        self.correction_tokenizer = AutoTokenizer.from_pretrained(correction_model_tag, use_auth_token=True)
        self.correction_model     = AutoModelForSeq2SeqLM.from_pretrained(correction_model_tag, use_auth_token=True)
        self.correction_model     = self.correction_model.to(device)
        self.model_loaded = True
        print("[Gramformer] Grammar error correct/highlight model loaded..")
    elif models == 2:
        # TODO
        print("TO BE IMPLEMENTED!!!")

  def correct(self, input_sentence, max_candidates=1):
      if self.model_loaded:
        correction_prefix = "gec: "
        input_sentence = correction_prefix + input_sentence
        input_ids = self.correction_tokenizer.encode(input_sentence, return_tensors='pt')
        input_ids = input_ids.to(self.device)

        preds = self.correction_model.generate(
            input_ids,
            do_sample=True, 
            max_length=128, 
#             top_k=50, 
#             top_p=0.95, 
            num_beams=7,
            early_stopping=True,
            num_return_sequences=max_candidates)

        corrected = set()
        for pred in preds:  
          corrected.add(self.correction_tokenizer.decode(pred, skip_special_tokens=True).strip())

        #corrected = list(corrected)
        #scores = self.scorer.sentence_score(corrected, log=True)
        #ranked_corrected = [(c,s) for c, s in zip(corrected, scores)]
        #ranked_corrected.sort(key = lambda x:x[1], reverse=True)
        return corrected
      else:
        print("Model is not loaded")  
        return None