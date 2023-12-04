from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_path = "/model/Whisper-large-v3"

processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)
model.config.forced_decoder_ids = None

ds = load_from_disk("/dataset/librispeech_asr_clean_test")

print("===========")
for i in range(5):
    sample = ds[i]["audio"]
    text = ds[i]["text"]
    print(text)
    input_features = processor(
        sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
    ).input_features

    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print(transcription[0].strip())
    print("===========")
