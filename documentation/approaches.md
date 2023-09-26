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

#### ideas
- RMSNorm (Zhang and Sennrich, 2019)
- SwiGLU activation function (Shazeer, 2020)
- grouped-query attention (GQA). 
- xformer.

### Second approach (Hard)

https://www.youtube.com/watch?v=4vGaN1dTVhw&ab_channel=Brainxyz


Questions
1. Consider already having a initial value, \
without increasing size of an embedding how to inject information \
in such a way that there is a relationship between them.
2. How can we inject new token without losing pretraining info?
3. How can we inject new category without losing pretraining info?
4. Is there any way to make the pretraining less expensive?\
5. <span style="color: red;"> Why pass the full function definition? Only function signature should be good enough.</span>

### Category Map Block

#### <span style="color: green;">Approach One</span>
- Calculate normal self attention
- Replace the function param tokens after calculating them using cross-attention
- Then apply feed forward

#### <span style="color: red;">Approach Two</span>
- Modify Multi head attention to accommodate the changes.
 