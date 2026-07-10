


Training diagnostics is the practice of inspecting a network's internal state to identify why it's not learning. Three signals tell you almost everything: activation statistics (is the signal dying or exploding as it flows forward?), gradient statistics (is the learning signal dying or exploding as it flows backward?), and the dead neuron fraction (are neurons permanently stuck at zero?).

The `diagnose` function applies a simple priority-ordered ruleset: dead neurons first (most severe), then exploding gradients, then vanishing gradients, then activation range checks.



![[Pasted image 20260710183958.png]]

## Dead Neuron
![[Pasted image 20260710184127.png]]


## Vanishing Gradients

![[Pasted image 20260710184314.png]]

***Early layers barely learn -> Weights get stuck***


## Exploding Gradients

![[Pasted image 20260710184539.png]]

### **Fix:** - Gradient Clipping, Proper Initialization, Normalization Layer


## Health Check

![[Pasted image 20260710184708.png]]

### **Healthy:-** Activation Std $\approx$ 0.5 - 1.0, low dead fractions, stable gradients


