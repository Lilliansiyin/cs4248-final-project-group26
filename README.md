# cs4248-final-project-group26

## **Abstract**
Our project aims to classify pre-defined relations(sister, friend, colleague...) between two arguments that appear in dialogue, based on an existing annotated dataset from “Friends”. We are going to use BERT and LSTM to achieve this task and use the confusion matrix and F1 score to analyze and evaluate the performance.  
  
The data set is retrieved from the original complete transcripts of *Friends* (All ten seasons 263 episodes in total with removing all content (usually in parentheses or square brackets) that describes non-verbal information such as behaviors and scene information.), from which to annotate all occurrences of 36 possible relation types
  
  
  

## **Approach**
Given dataset Dialogue RE, we will apply several deep learning sequential models such as LSTM， BiLSTM and BERT, to predict relations between pairs of entities in dialogues. Tasks are mainly on preprocessing dialogues, designing and building models, finetuning and comparing with others’ work.


## **User Guide**  
### **BERT**
All BERT trained model as well as related code work are attached in the below experiment files link.
Experiment files link: https://drive.google.com/drive/folders/18OsRMiWjNWifHeIteZKaDNT9GJC6rplT?usp=sharing

To Run prediction on test set, please run the following command:
  ```
  python run_classifier.py   --task_name berts  --do_train --do_eval   --data_dir . --vocab_file $BERT_DIR/vocab.txt   --bert_config_file $BERT_DIR/bert_config.json   --init_checkpoint $BERT_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir berts_f1  --gradient_accumulation_steps 2

  python evaluate.py --f1dev berts_f1/logits_dev.txt --f1test berts_f1/logits_test.txt --f1cdev berts_f1c/logits_dev.txt --f1ctest berts_f1c/logits_test.txt
  ```