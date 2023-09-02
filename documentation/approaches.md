### First approach (Easy)
- I don't need siamese network
- All token size same 768 only
  - Advantages: 
    - Simpler model
    - Similar tokenizer
    - Same length will ensure same category encoding
  - Drawbacks:
    - Relationship between functions
    - func-func translation functionality will degrade
    - Recursive/Combination of functions

### Second approach (Hard)

Questions

1. Consider already having a initial value, \
without increasing size of an embedding how to inject information \
in such a way that there is a relationship between them.
2. How can we inject new token without losing pretraining info?
3. How can we inject new category without losing pretraining info?
4. Is there any way to make the pretraining less expensive?