

![[Pasted image 20260708201019.png]]

![[Pasted image 20260708201029.png]]



![[Pasted image 20260708200940.png]]



**The Llama Architecture:** Meta's Llama, Llama 2, and Llama 3 all use RMSNorm instead of LayerNorm. So do Mistral, Gemma, and most modern open-source LLMs. When you see "Pre-RMSNorm" in an architecture diagram, it means RMSNorm is applied _before_ each attention and feed-forward sublayer (pre-norm style).

**Operation count comparison:**

| LayerNorm                | RMSNorm                           |                                    |
| ------------------------ | --------------------------------- | ---------------------------------- |
| **Mean computation**     | Yes                               | No                                 |
| **Variance computation** | Yes (needs mean first)            | Just mean of squares               |
| **Learnable parameters** | γγ and ββ (2n2n)                  | Only γγ (nn)                       |
| **Used in**              | Original Transformer, BERT, GPT-2 | Llama, Mistral, Gemma, modern LLMs |

**Why centering doesn't matter:** The original LayerNorm paper hypothesized that re-centering stabilizes training. The RMSNorm paper showed empirically that the scale invariance (dividing by RMS) is what actually helps. The mean subtraction adds cost without meaningful benefit.

RMS Normalization is the simplest normalization technique used in modern LLMs. It drops the two most complex parts of LayerNorm: the mean subtraction (re-centering) and the beta shift parameter. All that remains is dividing by the root mean square and scaling by gamma.

The insight from the RMSNorm paper: the re-centering (subtracting the mean) in LayerNorm is not what stabilizes training. The re-scaling (dividing by a magnitude measure) is. RMSNorm keeps only the re-scaling, using fewer operations and fewer parameters.