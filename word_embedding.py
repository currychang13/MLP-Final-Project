from gensim.models import Word2Vec

def update_skipgram_model(pretrained_path, new_patient_sequences):
    
    # 1. Load the pre-trained model
    # Note: You need the full model (not just KeyedVectors) to continue training
    model = Word2Vec.load(pretrained_path)
    
    print(f"Old vocab size: {len(model.wv)}")
    
    # 2. Update the Vocabulary
    # This adds new codes from your data to the model's dictionary
    model.build_vocab(new_patient_sequences, update=True)
    
    print(f"New vocab size: {len(model.wv)}")
    
    # 3. Fine-tune (Transfer Learning)
    # Train specifically on the new data. 
    # 'total_examples' and 'epochs' are critical here.
    model.train(new_patient_sequences, 
                total_examples=len(new_patient_sequences), 
                epochs=model.epochs)
                
    return model