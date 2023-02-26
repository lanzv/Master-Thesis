from src.models.nhpylm.model import Model

def accuracy(model: "Model", sentences_path: str, gold_path: str, dataset_title: str = "") -> float:
    with open(sentences_path) as f:
        sentences = f.read().splitlines()
    with open(gold_path) as f:
        segmented_sentences = f.read().splitlines()
    correct = 0
    all_sentences = 0
    for i, (sentence, segmented_sentence) in enumerate(zip(sentences, segmented_sentences)):
        predicted_segmentation = model.segment_sentence(sentence)
        all_sentences += 1
        #print("[",segmented_sentence, "],[", predicted_segmentation,"]")
        if predicted_segmentation == segmented_sentence:
            correct += 1
    print("{} Accuracy: {:.2f}% Correct: {:.2f}".format(dataset_title, (correct/all_sentences)*100, correct))
    return (correct/all_sentences)